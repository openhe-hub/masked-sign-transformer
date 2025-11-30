import argparse
import torch

def load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu")
    # 常见两种格式：直接是 state_dict，或者包在一个 dict 里
    if isinstance(ckpt, dict):
        # 优先尝试常见 key
        for key in ["state_dict", "model", "net", "module"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="path to .pth/.pt file")
    args = parser.parse_args()

    state_dict = load_state_dict(args.ckpt)

    total = 0
    print(f"== Parameter counts for {args.ckpt} ==")
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        n = tensor.numel()
        total += n
        print(f"{name:50s} : {n:10d}")

    print("-" * 70)
    print(f"Total params: {total}  ({total/1e6:.3f} M)")

if __name__ == "__main__":
    main()
