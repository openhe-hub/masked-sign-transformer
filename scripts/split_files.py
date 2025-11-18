import argparse
import math
import shutil
from pathlib import Path

def split_pkls(src_dir: Path, parts: int, mode: str) -> None:
        files = sorted(src_dir.glob("*.pkl"))
        if not files:
            raise SystemExit(f"No .pkl files found in {src_dir}")
        chunk = math.ceil(len(files) / parts)

        for part_idx in range(parts):
            dst_dir = src_dir.parent / f"{src_dir.name}_part{part_idx + 1:02d}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for p in files[part_idx * chunk: (part_idx + 1) * chunk]:
                dst = dst_dir / p.name
                if dst.exists():
                    raise SystemExit(
                        f"{dst} already exists; remove it or choose a different target folder.")
                if mode == "move":
                    shutil.move(p, dst)
                else:
                    shutil.copy2(p, dst)
            print(f"Part {part_idx + 1:02d}: stored files in {dst_dir}")

if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            description="Split .pkl files into equal parts.")
        parser.add_argument("--src-dir", default="output/bridge/phoenix",
                            type=Path, help="Directory with .pkl files.")
        parser.add_argument("--parts", default=10, type=int,
                            help="How many splits to create.")
        parser.add_argument("--mode", choices=("move", "copy"),
                            default="move", help="Move or copy files into the splits.")
        args = parser.parse_args()

        if args.parts < 1:
            parser.error("--parts must be ≥ 1.")
        split_pkls(args.src_dir, args.parts, args.mode)
