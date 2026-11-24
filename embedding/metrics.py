"""Retrieval and classification metrics for class-labelled embeddings.

Every item is used as a query against all the others (itself excluded), ranked by cosine
similarity; an item counts as relevant when it has the query's class.
"""
from collections import Counter

import numpy as np


class EmbeddingMetrics:
    """Metrics for embeddings X [N, D] with integer labels [N] indexing `classes`."""

    def __init__(self, X: np.ndarray, labels: np.ndarray, classes: list[str]):
        self.labels, self.classes = np.asarray(labels), classes
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        sim = Xn @ Xn.T
        np.fill_diagonal(sim, -np.inf)                       # a query never retrieves itself
        self.ranking = np.argsort(-sim, axis=1)[:, :-1]      # [N, N-1], most similar first
        self.match = self.labels[self.ranking] == self.labels[:, None]   # [N, N-1] relevant?

    def precision_at(self, k: int, per_query: bool = False):
        """P@k: fraction of the top-k results with the query's class."""
        p = self.match[:, :k].mean(axis=1)
        return p if per_query else float(p.mean())

    def recall_at(self, k: int, per_query: bool = False):
        """R@k (metric-learning "Recall@K"): fraction of queries with at least one hit in the top k."""
        r = self.match[:, :k].any(axis=1).astype(float)
        return r if per_query else float(r.mean())

    def mean_average_precision(self, per_query: bool = False):
        """mAP over each query's full ranking."""
        hits = self.match.cumsum(axis=1)
        precision = hits / np.arange(1, self.match.shape[1] + 1)
        n_rel = self.match.sum(axis=1)
        ap = np.where(n_rel > 0, (precision * self.match).sum(axis=1) / np.maximum(n_rel, 1), np.nan)
        return ap if per_query else float(np.nanmean(ap))

    def knn_predict(self, k: int = 5) -> np.ndarray:
        """Leave-one-out k-NN class prediction: majority of the k nearest, ties -> nearest."""
        top = self.labels[self.ranking[:, :k]]
        pred = np.empty(len(top), int)
        for i, row in enumerate(top):
            counts = Counter(row)
            best = max(counts.values())
            pred[i] = next(c for c in row if counts[c] == best)   # first = nearest among the tied
        return pred

    def knn_accuracy(self, k: int = 5) -> float:
        return float((self.knn_predict(k) == self.labels).mean())

    def per_class(self, values: np.ndarray) -> dict[str, float]:
        """Average a per-query metric (e.g. precision_at(10, per_query=True)) within each class."""
        return {self.classes[c]: float(np.nanmean(values[self.labels == c])) for c in np.unique(self.labels)}

    def confused_pairs(self, k: int = 5, top: int = 5) -> list[tuple[str, str, int]]:
        """Most frequent (true class, predicted class, count) mistakes of the k-NN classifier."""
        pred = self.knn_predict(k)
        pairs = Counter((self.classes[t], self.classes[p]) for t, p in zip(self.labels, pred) if t != p)
        return [(t, p, n) for (t, p), n in pairs.most_common(top)]

    def class_table(self, k: int = 10, knn_k: int = 5) -> list[dict]:
        """One row per class, worst P@k first: parts, P@1, P@k, R@5, AP, k-NN accuracy and
        the class its parts are most often mistaken for."""
        pred = self.knn_predict(knn_k)
        p1, pk = self.precision_at(1, True), self.precision_at(k, True)
        r5, ap = self.recall_at(5, True), self.mean_average_precision(True)
        rows = []
        for c in np.unique(self.labels):
            m = self.labels == c
            wrong = Counter(pred[m & (pred != c)])
            rows.append({"class": self.classes[c], "parts": int(m.sum()),
                         "P@1": float(p1[m].mean()), f"P@{k}": float(pk[m].mean()),
                         "R@5": float(r5[m].mean()), "AP": float(np.nanmean(ap[m])),
                         f"{knn_k}-NN acc": float((pred[m] == c).mean()),
                         "most confused with": self.classes[wrong.most_common(1)[0][0]] if wrong else ""})
        return sorted(rows, key=lambda r: r[f"P@{k}"])

    def print_class_table(self, k: int = 10, knn_k: int = 5, csv_path=None) -> None:
        """Print class_table(); with `csv_path`, also save it as CSV."""
        rows = self.class_table(k, knn_k)
        cols = list(rows[0])
        width = max(len(r["class"]) for r in rows)
        print(f"{'class':<{width}}  " + "  ".join(f"{c:>8}" for c in cols[1:-1]) + "  most confused with")
        for r in rows:
            nums = "  ".join(f"{r[c]:>8}" if isinstance(r[c], int) else f"{r[c]:>8.2f}" for c in cols[1:-1])
            print(f"{r['class']:<{width}}  {nums}  {r[cols[-1]]}")
        if csv_path:
            import csv
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=cols)
                writer.writeheader()
                writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
                                 for r in rows)
            print(f"saved {csv_path}")

    def summary(self, ks=(1, 5, 10), knn_k: int = 5) -> dict[str, float]:
        """Overall metrics: P@k and R@k for each k, mAP and k-NN accuracy."""
        out = {f"P@{k}": self.precision_at(k) for k in ks}
        out |= {f"R@{k}": self.recall_at(k) for k in ks}
        out["mAP"] = self.mean_average_precision()
        out[f"{knn_k}-NN acc"] = self.knn_accuracy(knn_k)
        return out

    def report(self, ks=(1, 5, 10), knn_k: int = 5, show: int = 5) -> None:
        """Print the overall metrics, best/worst classes by P@10 and the most confused pairs."""
        n_cls = len(np.unique(self.labels))
        print(f"{len(self.labels)} parts, {n_cls} classes (chance P@k ~ {1 / n_cls:.2f})")
        print("  " + "  ".join(f"{name} {v:.2f}" for name, v in self.summary(ks, knn_k).items()))
        if n_cls > 2:
            k = max(ks)
            per = sorted(self.per_class(self.precision_at(k, per_query=True)).items(), key=lambda kv: -kv[1])
            n = min(show, len(per) // 2)
            print(f"  best P@{k}:  " + ", ".join(f"{c} {v:.2f}" for c, v in per[:n]))
            print(f"  worst P@{k}: " + ", ".join(f"{c} {v:.2f}" for c, v in per[-n:][::-1]))
            for t, p, cnt in self.confused_pairs(knn_k, show):
                print(f"  {t} -> taken for {p}: {cnt}x")
