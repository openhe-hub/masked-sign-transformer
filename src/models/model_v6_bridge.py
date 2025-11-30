import torch
import torch.nn as nn

from config_loader import config
from utils.constants import PART_KP_INDICES, get_part_kp_indices


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding that supports batch_first inputs.
    """

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence length exceeds maximum positional encoding length.")
        return x + self.pe[:, : x.size(1)]


class PartBridgeModule(nn.Module):
    """
    Encoder-decoder block that predicts the gap segment for a single body part.
    """

    def __init__(self, n_kps, gap_length, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        input_dim = n_kps * 3
        self.n_kps = n_kps
        self.gap_length = gap_length

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.frame_type_embedding = nn.Embedding(2, hidden_dim)  # 0=pre, 1=post
        self.positional_encoder = SinusoidalPositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.gap_queries = nn.Parameter(torch.randn(gap_length, hidden_dim))
        self.gap_positional = SinusoidalPositionalEncoding(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, pre, post, pre_pad_mask, post_pad_mask):
        """
        Args:
            pre/post: tensors of shape (B, T_max, V, 3)
            pre_pad_mask/post_pad_mask: bool tensors (B, T_max) where True indicates padding.
        """
        B, T_pre, V, _ = pre.shape
        T_post = post.shape[1]

        pre_flat = pre.view(B, T_pre, -1)
        post_flat = post.view(B, T_post, -1)

        pre_feat = self.input_projection(pre_flat)
        post_feat = self.input_projection(post_flat)

        device = pre.device
        pre_type = torch.zeros(T_pre, dtype=torch.long, device=device)
        post_type = torch.ones(T_post, dtype=torch.long, device=device)
        pre_feat = pre_feat + self.frame_type_embedding(pre_type)[None, :, :]
        post_feat = post_feat + self.frame_type_embedding(post_type)[None, :, :]

        context = torch.cat([pre_feat, post_feat], dim=1)
        context_pad_mask = torch.cat([pre_pad_mask, post_pad_mask], dim=1)

        context = self.positional_encoder(context)
        encoded = self.context_encoder(context, src_key_padding_mask=context_pad_mask)

        gap_queries = self.gap_queries.unsqueeze(0).expand(B, -1, -1)
        gap_queries = self.gap_positional(gap_queries)

        decoded = self.decoder(
            tgt=gap_queries,
            memory=encoded,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=context_pad_mask,
        )

        output = self.output_head(decoded)
        return output.view(B, self.gap_length, self.n_kps, 3)


class PoseBridgeTransformerV6(nn.Module):
    """
    Part-aware bridge model that predicts the masked gap between two observed segments.
    """

    def __init__(self):
        super().__init__()
        model_cfg = config['model_v4']
        hidden_dim = model_cfg['hidden_dim']
        num_heads = model_cfg['num_heads']
        num_layers = model_cfg['num_layers']
        dropout = model_cfg['dropout']

        gap_length = config['masking']['gap_length']

        part_indices = get_part_kp_indices(include_face=True)
        self.parts = list(part_indices.keys())
        self.part_kps = {part: len(indices) for part, indices in part_indices.items()}

        self.part_modules = nn.ModuleDict(
            {
                part: PartBridgeModule(
                    n_kps=self.part_kps[part],
                    gap_length=gap_length,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                for part in self.parts
            }
        )

    def forward(self, pre_dict, post_dict, pre_pad_mask_dict, post_pad_mask_dict):
        outputs = {}
        for part in self.parts:
            outputs[part] = self.part_modules[part](
                pre_dict[part],
                post_dict[part],
                pre_pad_mask_dict[part],
                post_pad_mask_dict[part],
            )
        return outputs
