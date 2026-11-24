"""DINOv2 image embeddings for rendered CAD views (loaded with torch.hub, works on torch 2.4).

    python -m embedding.cad_embedder        # downloads the model once, prints output shapes

The first run downloads the DINOv2 code from GitHub and the weights (~350 MB for ViT-B/14)
into ~/.cache/torch/hub; later runs load them from there.
"""
import os

# DINOv2 resizes its position embeddings with bicubic interpolation, which torch 2.4 lacks on
# Apple GPUs (MPS); run just that op on the CPU. Must be set before torch is first imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import warnings

# Both expected and harmless: xFormers is an optional speed-up DINOv2 works without, and the
# CPU fallback above is the intended fix for the one bicubic op.
warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message=".*upsample_bicubic2d.*")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoCadImageEmbedder:
    """DINOv2 features for RGB uint8 images [H, W, 3]. OpenCV images are BGR: convert first,
    e.g. cv2.cvtColor(img, cv2.COLOR_BGR2RGB)."""

    def __init__(self, model_name: str = "dinov2_vitb14", size: int = 224, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        self.size = size  # must be a multiple of the 14 px patch size
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(self.device).eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def _batch(self, images: list[np.ndarray]) -> torch.Tensor:
        """List of RGB uint8 images -> normalised tensor [B, 3, size, size]. Resized, not
        cropped, because the renders are already framed on the part."""
        resized = [cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_AREA) for im in images]
        x = torch.from_numpy(np.stack(resized)).to(self.device).permute(0, 3, 1, 2).float() / 255
        return (x - self.mean) / self.std

    @torch.inference_mode()
    def global_embedding(self, images: list[np.ndarray]) -> np.ndarray:
        """One L2-normalised vector per image [B, 768] (CLS token): retrieval, classification."""
        return F.normalize(self.model(self._batch(images)), dim=-1).cpu().numpy()

    @torch.inference_mode()
    def patch_embeddings(self, images: list[np.ndarray]) -> np.ndarray:
        """Per-patch features [B, (size/14)^2, 768], e.g. a 16 x 16 grid at 224 px: dense matching."""
        return self.model.forward_features(self._batch(images))["x_norm_patchtokens"].cpu().numpy()

    def part_embedding(self, views: list[np.ndarray]) -> np.ndarray:
        """One vector for a whole part [768]: the mean of its views' embeddings, re-normalised.
        Averaging over cameras around the part makes it roughly independent of viewpoint."""
        mean = self.global_embedding(views).mean(axis=0)
        return mean / (np.linalg.norm(mean) + 1e-8)


if __name__ == "__main__":
    from cad_data_loader.cad_obj_dataset import MCB_ROOT, CadObjDataset
    from cad_data_loader.cad_views import create_views, load_triangles

    ds = CadObjDataset(MCB_ROOT / "train")
    index = ds.pick_index()
    views = [cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
             for v in create_views(load_triangles(ds.files[index]), views=8, size=224, label=False)]

    embedder = DinoCadImageEmbedder()
    print(f"{ds.classes[ds[index][1]]} / {ds.files[index].name}, {len(views)} views, on {embedder.device}")
    print("global_embedding", embedder.global_embedding(views).shape)
    print("patch_embeddings", embedder.patch_embeddings(views).shape)
    part = embedder.part_embedding(views)
    print("part_embedding  ", part.shape, f"norm {np.linalg.norm(part):.3f}")
