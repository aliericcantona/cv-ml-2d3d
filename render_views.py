"""Pre-render MCB parts to PNG views + index.csv for training (see cad_data_loader/view_dataset.py).

    python render_views.py                       # train and test, 16 views per part, ~11 GB
    python render_views.py --limit 50 --out /tmp/views_test     # quick try
"""
import argparse
from pathlib import Path

from cad_data_loader.cad_obj_dataset import MCB_ROOT, CadObjDataset
from cad_data_loader.view_dataset import render_split

if __name__ == "__main__":  # guard needed: render_split starts worker processes
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=MCB_ROOT.parent / "views", help="output folder")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--elevs", type=float, nargs="+", default=[20, 45], help="camera elevations, degrees")
    p.add_argument("--azims", type=int, default=8, help="cameras evenly spaced around each part")
    p.add_argument("--size", type=int, default=224, help="image width and height, px")
    p.add_argument("--workers", type=int, help="processes (default: all CPUs)")
    p.add_argument("--limit", type=int, help="only the first N parts per split")
    args = p.parse_args()

    classes = CadObjDataset(MCB_ROOT / "train").classes  # same label list for every split
    for split in args.splits:
        print(split)
        index = render_split(CadObjDataset(MCB_ROOT / split, classes=classes), args.out / split,
                             elevs=args.elevs, azims=[k * 360 / args.azims for k in range(args.azims)],
                             size=args.size, workers=args.workers, limit=args.limit)
        print(f"  wrote {index}")
