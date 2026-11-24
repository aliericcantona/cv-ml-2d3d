"""2D projections (PCA / MDS / t-SNE / UMAP) of class-labelled embeddings, and plots of them."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from umap import UMAP


class EmbeddingProjector:
    """2D projection of embeddings [N, D] with one of METHODS. Embeddings are L2-normalised;
    t-SNE and UMAP use cosine distance, MDS Euclidean (same ordering on unit vectors)."""

    METHODS = {
        "pca": lambda seed, n: PCA(n_components=2, random_state=seed),
        "mds": lambda seed, n: MDS(n_components=2, random_state=seed, n_init=1, init="classical_mds"),
        "tsne": lambda seed, n: TSNE(n_components=2, random_state=seed, metric="cosine",
                                     perplexity=min(30, (n - 1) / 3)),
        "umap": lambda seed, n: UMAP(n_components=2, random_state=seed, metric="cosine",
                                     n_neighbors=min(15, n - 1), n_jobs=1),  # seeded = 1 job
    }
    NAMES = {"pca": "PCA", "mds": "MDS", "tsne": "t-SNE", "umap": "UMAP"}

    def __init__(self, method: str = "tsne", seed: int = 42):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {list(self.METHODS)}")
        self.method, self.seed, self.name = method, seed, self.NAMES[method]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.METHODS[self.method](self.seed, len(X)).fit_transform(X)


class EmbeddingVisualizer:
    """Plots of 2D projections, one style (colour x marker) per class.

    thumbs: optional small RGB image per item, drawn instead of dots with `thumbs=True`."""

    MARKERS = ["o", "s", "^", "D"]  # x 20 colours = 80 distinct class styles
    LEGEND_ROWS = 24                # legend columns grow so each has at most this many rows

    def __init__(self, classes: list[str], thumbs: np.ndarray | None = None):
        self.classes, self.thumbs = classes, thumbs

    def style(self, c: int) -> tuple:
        """(colour, marker) for class index c."""
        return plt.cm.tab20(c % 20), self.MARKERS[(c // 20) % len(self.MARKERS)]

    def plot(self, ax, Y: np.ndarray, labels: np.ndarray, title: str = "",
             thumbs: bool = False, names: bool = False) -> None:
        """Scatter Y [N, 2] by class; `thumbs` draws each item's picture in a class-coloured
        frame, `names` writes each class name at the median of its points."""
        for c, cls in enumerate(self.classes):
            color, marker = self.style(c)
            m = labels == c
            ax.scatter(Y[m, 0], Y[m, 1], color=color, marker=marker, s=14, label=cls)
        if thumbs and self.thumbs is not None:
            zoom = (32 if len(Y) <= 300 else 20) / self.thumbs.shape[1]
            for (x, y), img, c in zip(Y, self.thumbs, labels):
                ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=zoom), (x, y), frameon=True, pad=0.05,
                                             bboxprops=dict(edgecolor=self.style(c)[0], linewidth=1.5)))
        if names:
            for c in np.unique(labels):
                x, y = np.median(Y[labels == c], axis=0)
                ax.annotate(self.classes[c][:22], (x, y), fontsize=6, ha="center", va="center",
                            color=self.style(c)[0], fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])

    def figure(self, X: np.ndarray, labels: np.ndarray, methods=("pca", "mds", "tsne", "umap"),
               thumbs: bool = False, names: bool = False, seed: int = 42):
        """One plot per projection method side by side, with a shared legend on the right."""
        size = 5 if len(self.classes) <= 20 else 8                   # inches per plot
        ncol = -(-len(self.classes) // self.LEGEND_ROWS)             # legend columns
        legend_w = 2.6 if ncol == 1 else 2.2 * ncol                  # inches for the legend
        # constrained layout makes room for the legend outside the plots (tight_layout doesn't)
        fig, axes = plt.subplots(1, len(methods), squeeze=False, layout="constrained",
                                 figsize=(size * len(methods) + legend_w, size + 0.3))
        for ax, method in zip(axes[0], methods):
            projector = EmbeddingProjector(method, seed)
            self.plot(ax, projector.fit_transform(X), labels, projector.name, thumbs, names)
        handles, texts = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, texts, loc="outside right center", frameon=False, ncol=ncol,
                   fontsize=8 if ncol == 1 else 6.5, markerscale=1.3, handletextpad=0.2)
        return fig
