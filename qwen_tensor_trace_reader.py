import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect Qwen tensor trace NPY files")
    parser.add_argument("tensor_dir", help="Directory created by --tensor-mode npy")
    parser.add_argument("--pattern", default="*.npy", help="Glob pattern relative to tensor_dir")
    parser.add_argument("--sample-size", type=int, default=8, help="Number of flattened values to print")
    parser.add_argument("--list", action="store_true", help="List matching files without loading them")
    return parser.parse_args()


def describe(path, root, sample_size):
    array = np.load(path)
    flat = array.reshape(-1)
    print(f"{path.relative_to(root)}")
    print(f"  shape={array.shape}, dtype={array.dtype}, numel={array.size}")
    if flat.size == 0:
        print("  values=empty")
        return

    values = flat[:sample_size].astype(np.float32, copy=False).tolist()
    stats = flat.astype(np.float32, copy=False)
    sentinel = np.finfo(np.float32).min / 2
    valid = stats[np.isfinite(stats) & (stats > sentinel)]
    masked_or_nonfinite = stats.size - valid.size
    if valid.size == 0:
        print(f"  no valid values, masked_or_nonfinite={masked_or_nonfinite}")
    else:
        print(
            f"  valid_min={valid.min():.6g}, valid_max={valid.max():.6g}, "
            f"valid_mean={valid.mean():.6g}, masked_or_nonfinite={masked_or_nonfinite}"
        )
    print(f"  sample={values}")


def main():
    args = parse_args()
    root = Path(args.tensor_dir)
    if not root.is_dir():
        raise ValueError(f"tensor directory does not exist: {root}")

    files = sorted(root.rglob(args.pattern))
    if not files:
        raise ValueError(f"no files match {args.pattern!r} under {root}")

    for path in files:
        if args.list:
            print(path.relative_to(root))
        else:
            describe(path, root, args.sample_size)


if __name__ == "__main__":
    main()
