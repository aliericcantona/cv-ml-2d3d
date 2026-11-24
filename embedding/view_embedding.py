"""Embed MCB parts with DINOv2, print retrieval metrics and plot 2D projections by class.

    source setup.sh
    python embedding/view_embedding.py                          # 8 classes, all four methods
    python embedding/view_embedding.py --methods tsne --thumbs  # part pictures instead of dots
    python embedding/view_embedding.py --classes Springs Wheel Hinge --per-class 40
    python embedding/view_embedding.py --classes 15 --seed 1    # 15 random classes
    python embedding/view_embedding.py --classes all --methods tsne --cache all68.npz --names
    python embedding/view_embedding.py --classes all --cache all68.npz --report report.html
    python embedding/view_embedding.py --classes all --cache all68.npz --report classes.csv

Each point is one part: the average DINOv2 embedding of `--views` renders around it.
--cache saves the embeddings, so re-plotting skips rendering + DINO (all 68 classes: ~6 min).
"""
# part_embeddings first: it imports cad_embedder, which must come before torch
from embedding.part_embeddings import DEFAULT_CLASSES, describe_experiment, load_or_embed, pick_classes

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from cad_data_loader.cad_obj_dataset import MCB_ROOT, CadObjDataset
from embedding.html_report import HtmlReport
from embedding.metrics import EmbeddingMetrics
from embedding.visualizer import EmbeddingProjector, EmbeddingVisualizer

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--methods", nargs="+", default=list(EmbeddingProjector.METHODS),
               choices=list(EmbeddingProjector.METHODS))
p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
               help='class folder names, "all", or a number N for N random classes')
p.add_argument("--seed", type=int, default=42, help="which random classes for --classes N")
p.add_argument("--per-class", type=int, default=25, help="parts per class")
p.add_argument("--views", type=int, default=8, help="renders averaged per part")
p.add_argument("--split", default="train", choices=["train", "test"])
p.add_argument("--thumbs", action="store_true", help="draw part pictures instead of dots")
p.add_argument("--names", action="store_true", help="write class names on the plot")
p.add_argument("--cache", type=Path, metavar="PATH.npz", help="save / reuse the embeddings")
p.add_argument("--save", metavar="PATH.png", help="save the figure (no window)")
p.add_argument("--report", nargs="?", const="", metavar="PATH.html|PATH.csv",
               help="print the per-class table; with a .html path also write a report page "
                    "(metrics + plot + tables), with a .csv path save the table")
args = p.parse_args()

ds = CadObjDataset(MCB_ROOT / args.split)
classes = pick_classes(ds, args.classes, args.seed)
X, labels, thumbs = load_or_embed(args.cache, ds, classes, args.per_class, args.views)

metrics = EmbeddingMetrics(X, labels, classes)
metrics.report()
if args.report is not None:
    metrics.print_class_table(csv_path=args.report if args.report.endswith(".csv") else None)

fig = EmbeddingVisualizer(classes, thumbs).figure(X, labels, args.methods, args.thumbs, args.names)
if args.report and args.report.endswith(".html"):
    experiment = describe_experiment(ds, classes, args.classes, args.per_class, args.views, args.seed, len(X))
    path = HtmlReport().save(args.report, metrics, fig, "MCB part embeddings", experiment, args.methods)
    print(f"saved {path}")
if args.save:
    fig.savefig(args.save, dpi=110, bbox_inches="tight")
else:
    plt.show()
