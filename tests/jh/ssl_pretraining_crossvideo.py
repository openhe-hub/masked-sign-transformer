"""
Stage-1 SSL Pretraining with Cross-Video Contrastive Learning

Key features:
1. Transformer encoder (not ST-GCN)
2. Cross-video contrastive loss (same gloss = positive, different gloss = negative)
3. BalancedGlossSampler to ensure positive pairs in each batch
4. Progressive training: warmup without CL, then add CL with decay
5. Temporal convolution for reduced representation density

Designed for isolated sign datasets (one video = one gloss).
"""

# --- injected fix: define gpu/local_rank from launcher env ---
import os as _os
gpu = int(_os.environ.get('LOCAL_RANK', _os.environ.get('SLURM_LOCALID', '0')))
# --------------------------------------------------------------
import os
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

import utils as utils
from datasets import S2T_Dataset
from ssl_models_crossvideo import PoseBlockEncoderSignCL
from config import train_label_paths, dev_label_paths


def create_blocks(tensor_btv3: torch.Tensor, K: int, stride: int):
    """
    tensor_btv3: [B, T, V, 3]
    Returns blocks of shape [B, Bk, K, V, 3] where Bk is number of blocks.
    """
    B, T, V, C = tensor_btv3.shape
    if T < K:
        return None
    idx = torch.arange(0, T - K + 1, stride, device=tensor_btv3.device)
    blocks = torch.stack([tensor_btv3[:, i:i+K] for i in idx.tolist()], dim=1)
    return blocks


def random_mask(shape, mask_ratio_low=0.2, mask_ratio_high=0.3, device='cpu'):
    B, Bk, K, V = shape
    ratio = torch.empty((B, Bk, 1, 1), device=device).uniform_(mask_ratio_low, mask_ratio_high)
    rand = torch.rand((B, Bk, K, V), device=device)
    return (rand < ratio).bool()


class BalancedGlossSampler(Sampler):
    """
    Sampler that ensures each batch contains multiple videos of the same gloss.

    This is critical for cross-video contrastive learning, which requires
    positive pairs (same gloss) within each batch.

    Strategy:
    - Group dataset by gloss label
    - Sample glosses, then sample multiple videos per gloss
    - Each batch has samples_per_gloss videos of the same gloss

    Args:
        dataset: S2T_Dataset instance
        batch_size: Total batch size
        samples_per_gloss: Number of videos per gloss in each batch (default: 4)
        drop_last: Drop last incomplete batch
    """
    def __init__(self, dataset, batch_size, samples_per_gloss=4, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_gloss = samples_per_gloss
        self.drop_last = drop_last

        # Build gloss -> indices mapping
        self.gloss_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            # Get gloss from dataset
            sample = dataset[idx]
            # sample structure: (uid, pose_sample, gloss_seq, text, conf_sample)
            gloss_seq = sample[2]

            # For isolated signs, gloss_seq is a list with one element or a single value
            if isinstance(gloss_seq, (list, tuple)):
                gloss_label = gloss_seq[0] if len(gloss_seq) > 0 else 0
            else:
                gloss_label = gloss_seq

            self.gloss_to_indices[gloss_label].append(idx)

        # Keep only glosses with at least samples_per_gloss samples
        self.glosses = [g for g, indices in self.gloss_to_indices.items()
                       if len(indices) >= samples_per_gloss]

        print(f"[BalancedGlossSampler] Total glosses: {len(self.gloss_to_indices)}")
        print(f"[BalancedGlossSampler] Glosses with >={samples_per_gloss} samples: {len(self.glosses)}")

        # Calculate number of batches
        num_gloss_groups = len(self.glosses)
        self.num_batches = num_gloss_groups if self.drop_last else (num_gloss_groups + self.batch_size - 1) // self.batch_size

        print(f"[BalancedGlossSampler] Batches per epoch: {self.num_batches}")

    def __iter__(self):
        # Shuffle glosses
        glosses = torch.randperm(len(self.glosses)).tolist()

        batch = []
        for gloss_idx in glosses:
            gloss = self.glosses[gloss_idx]
            indices = self.gloss_to_indices[gloss]

            # Sample samples_per_gloss videos for this gloss
            if len(indices) >= self.samples_per_gloss:
                sampled = torch.randperm(len(indices))[:self.samples_per_gloss].tolist()
                batch.extend([indices[i] for i in sampled])

            # Yield batch when full
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

        # Yield remaining samples if not drop_last
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        """Set epoch for distributed training (changes random seed)"""
        torch.manual_seed(epoch)


def collate_blocks(batch, K: int, stride: int, device: str, vocab: dict):
    """
    Collate batch into blocks and extract gloss labels.

    Returns:
        inputs_by_part, masks, confs, labels
    """
    pose_batches = {k: [] for k in ['body', 'left', 'right', 'face_all']}
    conf_batches = {k: [] for k in ['body', 'left', 'right', 'face_all']}
    labels = []

    max_blocks = None
    for uid, pose_sample, gloss_seq, _, _ in batch:
        # Extract gloss label
        if isinstance(gloss_seq, (list, tuple)):
            gloss_token = gloss_seq[0] if len(gloss_seq) > 0 else '<unk>'
        else:
            gloss_token = gloss_seq

        # Convert gloss token to integer label
        label = vocab.get(gloss_token, vocab.get('<unk>', 1))
        labels.append(label)

        for part in pose_batches.keys():
            x = pose_sample[part].float()  # [T, N, 3]
            xb = create_blocks(x.unsqueeze(0), K, stride)
            if xb is None:
                continue
            xb = xb.squeeze(0)
            pose_batches[part].append(xb)
            conf_batches[part].append(xb[..., 2])
            max_blocks = xb.shape[0] if max_blocks is None else min(max_blocks, xb.shape[0])

    if max_blocks is None:
        raise ValueError("No samples produced any pose blocks.")

    # Align number of blocks
    for part in pose_batches.keys():
        pose_batches[part] = [xb[:max_blocks] for xb in pose_batches[part]]
        conf_batches[part] = [cb[:max_blocks] for cb in conf_batches[part]]

    # Stack -> [B, Bk, K, V, 3]
    inputs_by_part = {part: torch.stack(pose_batches[part], dim=0).to(device) for part in pose_batches}
    confs_by_part = {part: torch.stack(conf_batches[part], dim=0).to(device) for part in conf_batches}

    # Generate random masks
    masks_by_part = {part: random_mask(inputs_by_part[part][..., 0].shape, device=device) for part in inputs_by_part}

    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    return inputs_by_part, masks_by_part, confs_by_part, labels


def main(args):
    # Init distributed
    env_backend = os.environ.get("FORCE_DIST_BACKEND")
    if env_backend:
        args.dist_backend = env_backend

    utils.init_distributed_mode_ds(args)
    print(args)
    utils.set_seed(args.seed)

    # Dataset
    dataset = S2T_Dataset(path=train_label_paths[args.dataset], args=args, phase='train')

    # Load vocabulary for gloss labels
    import json
    vocab_path = f'data/{args.dataset}/vocab.json'
    print(f"Loading vocabulary from {vocab_path}")
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    vocab = vocab_data['token_to_id']  # Use token_to_id mapping
    print(f"Vocabulary size: {len(vocab)}")

    # Samplers & DataLoaders
    # CRITICAL: Always use BalancedGlossSampler for cross-video CL (even in distributed mode)
    if args.distributed:
        print(f"[WARNING] Using BalancedGlossSampler in distributed mode.")
        print(f"[WARNING] This may cause uneven data distribution across GPUs.")

    train_sampler = BalancedGlossSampler(
        dataset,
        batch_size=args.batch_size,
        samples_per_gloss=args.samples_per_gloss,
        drop_last=True
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=train_sampler,  # Use batch_sampler, not sampler
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=args.pin_mem,
    )

    eval_dataset = dataset
    if args.distributed:
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
    else:
        eval_sampler = None

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=eval_dataset.collate_fn,
        sampler=eval_sampler,
        shuffle=False,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cpu')

    # Create model with Transformer + Cross-Video CL
    model = PoseBlockEncoderSignCL(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        projection_dim=args.projection_dim,
        use_conf_weight=True
    ).to(device)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpu] if device.type == 'cuda' else None,
            output_device=gpu if device.type == 'cuda' else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if utils.is_main_process():
        print(f'Number of params: {n_parameters / 1e6:.2f}M')

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    # LR scheduler (cosine with warmup)
    total_steps = args.epochs * len(dataloader)
    warmup_steps = int(args.warmup_epochs * len(dataloader))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP
    use_amp = (args.dtype in ['fp16', 'bf16'])
    amp_dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and args.dtype == 'fp16'))

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir and utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    if utils.is_main_process():
        print(f"Start training for {args.epochs} epochs")
        print(f"Warmup epochs (no CL): {args.cl_warmup_epochs}")
        print(f"CL weight: {args.cl_weight}, Decay: {args.cl_decay_rate} every {args.cl_decay_interval} epochs")

    best_metric = float('inf')
    global_step = 0
    cl_weight = args.cl_weight
    last_decay_epoch = 0

    for epoch in range(args.epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # CL decay schedule (like SignCL paper)
        if epoch > last_decay_epoch + args.cl_decay_interval:
            cl_weight = cl_weight * args.cl_decay_rate
            last_decay_epoch = epoch
            if utils.is_main_process():
                print(f"Epoch {epoch}: Decayed CL weight to {cl_weight:.6f}")

        model.train()

        # Training loop
        for batch in (
            utils.MetricLogger(delimiter="  ").log_every(
                dataloader, 10, header=f"Epoch: [{epoch}/{args.epochs}]"
            )
        ):
            src_input, tgt_input = batch
            B = src_input['body'].shape[0]

            # Extract gloss labels from batch (already collated by dataset.collate_fn)
            gloss_batch = tgt_input['gt_gloss']  # List of gloss sequences

            # Convert gloss labels to tensor of integers
            labels_list = []
            for gloss_seq in gloss_batch:
                # For isolated signs, gloss_seq is a list with one element
                if isinstance(gloss_seq, (list, tuple)):
                    gloss_token = gloss_seq[0] if len(gloss_seq) > 0 else '<unk>'
                else:
                    gloss_token = gloss_seq

                # Convert gloss token to integer label
                label = vocab.get(gloss_token, vocab.get('<unk>', 1))
                labels_list.append(label)

            labels_batch = torch.tensor(labels_list, dtype=torch.long, device=device)

            def to_btv3(key):
                return src_input[key].to(torch.float32).to(device)

            inputs_all = {part: to_btv3(part) for part in ['body', 'left', 'right', 'face_all']}

            # Build blocks (no need to extract labels again - we have them from batch)
            inputs_by_part, masks_by_part, confs_by_part = {}, {}, {}

            # Create blocks for each sample
            all_blocks = {part: [] for part in ['body', 'left', 'right', 'face_all']}
            all_confs = {part: [] for part in ['body', 'left', 'right', 'face_all']}
            max_blocks = None

            for i in range(B):
                for part in ['body', 'left', 'right', 'face_all']:
                    x = inputs_all[part][i].float()  # [T, V, 3]
                    xb = create_blocks(x.unsqueeze(0), K=args.block_size, stride=args.block_stride)
                    if xb is None:
                        continue
                    xb = xb.squeeze(0)  # [Bk, K, V, 3]
                    all_blocks[part].append(xb)
                    all_confs[part].append(xb[..., 2])
                    max_blocks = xb.shape[0] if max_blocks is None else min(max_blocks, xb.shape[0])

            # Align number of blocks
            for part in all_blocks.keys():
                all_blocks[part] = [xb[:max_blocks] for xb in all_blocks[part]]
                all_confs[part] = [cb[:max_blocks] for cb in all_confs[part]]

            # Stack -> [B, Bk, K, V, 3]
            inputs_by_part = {part: torch.stack(all_blocks[part], dim=0).to(device) for part in all_blocks}
            confs_by_part = {part: torch.stack(all_confs[part], dim=0).to(device) for part in all_confs}

            # Generate random masks
            masks_by_part = {part: random_mask(inputs_by_part[part][..., 0].shape, device=device) for part in inputs_by_part}

            # Merge B and Bk for efficient compute
            def merge_blocks(x):
                B0, Bk, K, V, C = x.shape
                return x.view(B0 * Bk, K, V, C)

            def merge_mask(x):
                B0, Bk, K, V = x.shape
                return x.view(B0 * Bk, K, V)

            inputs_mb = {k: merge_blocks(v) for k, v in inputs_by_part.items()}
            masks_mb = {k: merge_mask(v) for k, v in masks_by_part.items()}
            confs_mb = {k: merge_mask(v) for k, v in confs_by_part.items()}

            # Get batch dimensions for video-level CL
            # B0 = number of videos, Bk = number of blocks per video
            B0, Bk = inputs_by_part['body'].shape[:2]

            # Progressive training: no CL during warmup
            compute_cl = (epoch >= args.cl_warmup_epochs)

            model.train()
            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                out = model(
                    inputs_mb, masks_mb, confs_mb,
                    labels=labels_batch,  # VIDEO-level labels [B0], not block-level!
                    num_videos=B0,  # Tell model to aggregate blocks → videos
                    reconstruct_xy_only=True,
                    compute_cl_loss=compute_cl,
                    temperature=args.temperature,
                    conf_threshold=args.conf_threshold  # Confidence-based masking
                )

                # Weight CL loss (like SignCL: 0.001 * cl_loss)
                if compute_cl:
                    loss = out['recon_loss'] + cl_weight * out['cl_loss']
                else:
                    loss = out['recon_loss']

                loss = loss / max(1, args.gradient_accumulation_steps)

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % max(1, args.gradient_accumulation_steps) == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_without_ddp.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model_without_ddp.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            global_step += 1
            if utils.is_main_process() and (global_step % 50 == 0):
                recon_loss_val = out['recon_loss'].item()
                cl_loss_val = out['cl_loss'].item() if compute_cl else 0.0
                print(f"step {global_step} total_loss {loss.item():.6f} "
                      f"recon {recon_loss_val:.6f} cl {cl_loss_val:.6f} "
                      f"lr {optimizer.param_groups[0]['lr']:.6e}")

        # Synchronize before checkpointing
        if args.distributed:
            dist.barrier()

        # Save checkpoint every epoch
        if args.output_dir and utils.is_main_process():
            ckpt = {
                'model': model_without_ddp.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'cl_weight': cl_weight,
            }
            try:
                torch.save(ckpt, output_dir / f'ssl_checkpoint_{epoch}.pth')
                print(f"Saved checkpoint for epoch {epoch}")
            except Exception as e:
                print(f"Warning: Failed to save checkpoint for epoch {epoch}: {e}")

        # Eval (only reconstruction, no CL)
        metric = evaluate_ssl(args, model_without_ddp, eval_loader, device, vocab)
        if utils.is_main_process():
            print({k: float(v) for k, v in metric.items()})
            current = float(metric.get('masked_mse_xy', float('inf')))
            if current < best_metric:
                best_metric = current
                best_ckpt = {
                    'model': model_without_ddp.state_dict(),
                    'args': vars(args),
                    'epoch': epoch,
                    'best_metric': best_metric,
                    'cl_weight': cl_weight,
                }
                try:
                    torch.save(best_ckpt, output_dir / 'best.pth')
                    print(f"Saved best checkpoint (metric: {best_metric:.6f})")
                except Exception as e:
                    print(f"Warning: Failed to save best checkpoint: {e}")

    if utils.is_main_process():
        print("Training complete!")


def evaluate_ssl(args, model, eval_loader, device, vocab):
    """Evaluation: only reconstruction loss."""
    model.eval()
    total_mse = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in eval_loader:
            src_input, _ = batch
            B = src_input['body'].shape[0]

            def to_btv3(key):
                return src_input[key].to(torch.float32).to(device)

            inputs_all = {part: to_btv3(part) for part in ['body', 'left', 'right', 'face_all']}

            inputs_by_part, masks_by_part, confs_by_part, labels_batch = collate_blocks(
                [(None, {p: inputs_all[p][i] for p in inputs_all}, None, None, None) for i in range(B)],
                K=args.block_size,
                stride=args.block_stride,
                device=device,
                vocab=vocab
            )

            def merge_blocks(x):
                B0, Bk, K, V, C = x.shape
                return x.view(B0 * Bk, K, V, C)

            def merge_mask(x):
                B0, Bk, K, V = x.shape
                return x.view(B0 * Bk, K, V)

            inputs_mb = {k: merge_blocks(v) for k, v in inputs_by_part.items()}
            masks_mb = {k: merge_mask(v) for k, v in masks_by_part.items()}
            confs_mb = {k: merge_mask(v) for k, v in confs_by_part.items()}

            out = model(
                inputs_mb, masks_mb, confs_mb,
                labels=None,  # No CL loss during eval
                reconstruct_xy_only=True,
                compute_cl_loss=False
            )

            total_mse += out['recon_loss'].item()
            total_count += 1

    return {'masked_mse_xy': total_mse / max(1, total_count)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SSL pretraining with SignCL')
    parser.add_argument('--dataset', default='ASL', type=str)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--warmup-epochs', default=5, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)

    # Model architecture (Transformer-based)
    parser.add_argument('--hidden-dim', default=256, type=int, help='Transformer hidden dimension')
    parser.add_argument('--num-heads', default=8, type=int, help='Transformer attention heads')
    parser.add_argument('--num-layers', default=4, type=int, help='Transformer encoder layers')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')

    # Cross-Video CL loss
    parser.add_argument('--projection-dim', default=128, type=int, help='Projection head output dim')
    parser.add_argument('--temperature', default=0.07, type=float, help='InfoNCE temperature')
    parser.add_argument('--samples-per-gloss', default=4, type=int, help='Videos per gloss in batch')
    parser.add_argument('--cl-weight', default=0.5, type=float, help='CL loss weight (higher than SignCL)')
    parser.add_argument('--cl-warmup-epochs', default=10, type=int, help='Epochs before enabling CL')
    parser.add_argument('--cl-decay-rate', default=0.95, type=float, help='CL weight decay rate')
    parser.add_argument('--cl-decay-interval', default=20, type=int, help='Epochs between CL weight decays')

    # Reconstruction loss
    parser.add_argument('--conf-threshold', default=0.3, type=float,
                       help='Confidence threshold for valid joints (reconstruction masking)')

    # Block params
    parser.add_argument('--block-size', default=20, type=int, help='K frames per block')
    parser.add_argument('--block-stride', default=10, type=int, help='Frame stride between blocks')

    # System
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin-mem', action='store_true', default=True)
    parser.add_argument('--dtype', default='fp16', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--seed', default=42, type=int)

    # Distributed
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--dist-url', default='env://', type=str)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)

    args = parser.parse_args()

    # Add defaults required by S2T_Dataset
    args.rgb_support = False
    args.max_length = 256

    main(args)