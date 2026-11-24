"""Compute (and cache) one DINO embedding per MCB part, for analysis and plotting."""
# cad_embedder first: it sets the MPS fallback, which must happen before torch is imported
from embedding.cad_embedder import DinoCadImageEmbedder

import random
from pathlib import Path

import cv2
import numpy as np

from cad_data_loader.cad_obj_dataset import CadObjDataset
from cad_data_loader.cad_views import create_views, load_triangles

DEFAULT_CLASSES = ["Right spur gears", "Hexagonal nuts", "Springs", "Wheel", "Hinge",
                   "Impeller", "Flanged block bearing", "Screws and bolts with hexagonal head"]
THUMB_PX = 48


def pick_classes(ds: CadObjDataset, spec: list[str], seed: int = 42) -> list[str]:
    """["all"] -> every class, ["15"] -> 15 random classes (fixed by `seed`), else the names given."""
    if spec == ["all"]:
        return list(ds.classes)
    if len(spec) == 1 and spec[0].isdigit():
        return sorted(random.Random(seed).sample(ds.classes, int(spec[0])))
    unknown = [c for c in spec if c not in ds.class_to_idx]
    if unknown:
        raise ValueError(f"unknown classes {unknown}; choose from: {', '.join(ds.classes)}")
    return list(spec)


def embed_parts(ds: CadObjDataset, embedder: DinoCadImageEmbedder, classes: list[str],
                per_class: int = 25, views: int = 8, seed: int = 42):
    """Embed `per_class` random parts from each class folder (fewer if a class is smaller).
    Returns embeddings [N, D], labels [N] (index into `classes`) and a small RGB thumbnail per part."""
    rng = random.Random(seed)
    X, labels, thumbs = [], [], []
    for c, cls in enumerate(classes):
        indices = [i for i, f in enumerate(ds.files) if f.parent.name == cls]
        for i in rng.sample(indices, min(per_class, len(indices))):
            rgb = [cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
                   for v in create_views(load_triangles(ds.files[i]), views=views, size=224, label=False)]
            X.append(embedder.part_embedding(rgb))
            labels.append(c)
            thumbs.append(cv2.resize(rgb[0], (THUMB_PX, THUMB_PX), interpolation=cv2.INTER_AREA))
        print(f"  [{c + 1}/{len(classes)}] {cls}: {labels.count(c)} parts", flush=True)
    return np.stack(X), np.array(labels), np.stack(thumbs)


def describe_experiment(ds: CadObjDataset, classes: list[str], spec: list[str], per_class: int,
                        views: int, seed: int, n_parts: int) -> dict[str, str]:
    """Label -> HTML description of how the embeddings were made (for the HTML report).
    Mirrors embed_parts(): rendering uses cad_views.create_views defaults (25 deg, 224 px, 40 deg FOV)."""
    if spec == ["all"]:
        chosen = f"all {len(classes)} classes"
    elif len(spec) == 1 and spec[0].isdigit():
        chosen = f"{len(classes)} random classes (seed {seed})"
    elif classes == DEFAULT_CLASSES:
        chosen = f"{len(classes)} hand-picked, visually distinct classes (default)"
    else:
        chosen = f"{len(classes)} chosen classes"
    return {
        "Dataset": f"MCB (Mechanical Components Benchmark), subset <b>{ds.root.parent.name}</b>, "
                   f"<b>{ds.root.name}</b> split: {len(ds.files):,} parts in {len(ds.classes)} classes (one folder per class)",
        "Sample": f"{chosen}, up to {per_class} random parts each (seed {seed}): <b>{n_parts} parts</b>",
        "Classes": ", ".join(classes),
        "Rendering": f"{views} views per part: pinhole camera at 25&deg; elevation, every {360 / views:g}&deg; "
                     f"of azimuth, 224&times;224 px, 40&deg; field of view; part centred and framed, flat shading "
                     f"(<code>cad_views.create_views</code>, OpenCV)",
        "Embedding model": "DINOv2 ViT-B/14 (self-supervised, no fine-tuning), via <code>torch.hub</code>; "
                           "ImageNet normalisation; CLS token &rarr; 768-d view embedding, L2-normalised",
        "Part embedding": f"mean of the {views} view embeddings, re-normalised to length 1 "
                          f"(<code>DinoCadImageEmbedder.part_embedding</code>)",
        "Similarity": "cosine similarity between part embeddings",
    }


def load_or_embed(cache: Path | None, ds: CadObjDataset, classes: list[str], per_class: int = 25,
                  views: int = 8):
    """embed_parts(), reusing `cache` (.npz) when it was made with the same settings."""
    settings = dict(classes=classes, per_class=per_class, views=views, split=ds.root.name)
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        if z["settings"].item() == settings:
            print(f"embeddings from {cache}")
            return z["X"], z["labels"], z["thumbs"]
        print(f"{cache} was made with other settings, recomputing")
    X, labels, thumbs = embed_parts(ds, DinoCadImageEmbedder(), classes, per_class, views)
    if cache:
        np.savez_compressed(cache, X=X, labels=labels, thumbs=thumbs,
                            settings=np.array(settings, dtype=object))
        print(f"saved {cache}")
    return X, labels, thumbs
