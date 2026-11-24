"""Fill report_template.html with embedding metrics and a matplotlib figure: one standalone file.

Template placeholders (string.Template), each filled by HtmlReport.render():
    $title $subtitle $cards $experiment $methods $figure $confused_rows $class_head $class_rows $generated
string.Template substitutes them everywhere in the template, comments included, so don't
mention them in the template itself. A literal dollar sign in the template must be written $$.
"""
import base64
import io
from datetime import datetime
from html import escape
from pathlib import Path
from string import Template

from embedding.metrics import EmbeddingMetrics

TEMPLATE = Path(__file__).with_name("report_template.html")

METHOD_INFO = {
    "pca": ("PCA", "Linear: the two directions of largest variance. Keeps the global layout, but two of 768 dimensions usually overlap the classes."),
    "mds": ("MDS", "Places points so all pairwise distances (near and far) match as well as possible. In high dimensions most distances are similar, so it tends to give a blurred ring."),
    "tsne": ("t-SNE", "Keeps each point's nearest neighbours (perplexity 30, cosine distance). Clusters are meaningful; distances between clusters and cluster sizes are not. Cannot place new points."),
    "umap": ("UMAP", "Keeps nearest neighbours (15, cosine) plus some global structure; tighter clusters than t-SNE and can place new points. Outliers can squeeze the rest of the plot."),
}
METRIC_INFO = {
    "P": "precision at k: fraction of the k most similar parts with the query's class",
    "R": "Recall@K: fraction of queries with at least one same-class part in the top k",
    "mAP": "mean average precision over each query's full ranking",
    "NN": "leave-one-out k-NN classification accuracy (majority of the k nearest)",
}


class HtmlReport:
    """Renders the template's $placeholders; the figure is embedded as a PNG data URI."""

    def __init__(self, template: Path = TEMPLATE):
        self.template = Template(Path(template).read_text())

    @staticmethod
    def figure_uri(fig, dpi: int = 110) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _cards(summary: dict[str, float], chance: float) -> str:
        def tip(name):
            key = "NN" if "NN" in name else name if name == "mAP" else name[0]
            return escape(f"{name}: {METRIC_INFO.get(key, '')}")
        cards = [f'<div class="card" title="{tip(k)}"><div class="name">{escape(k)}</div>'
                 f'<div class="value">{v:.2f}</div></div>' for k, v in summary.items()]
        cards.append(f'<div class="card chance" title="1 / number of classes"><div class="name">chance P@k</div><div class="value">{chance:.2f}</div></div>')
        return "\n".join(cards)

    @staticmethod
    def _facts(items: dict) -> str:
        """<dt>/<dd> pairs; values are trusted HTML (they come from our own code)."""
        return "".join(f"<dt>{escape(str(k))}</dt><dd>{v}</dd>" for k, v in items.items())

    @staticmethod
    def _class_table(rows: list[dict]) -> tuple[str, str]:
        cols = list(rows[0])
        head = "".join(f'<th class="text">{escape(c)}</th>' if isinstance(rows[0][c], str) else f"<th>{escape(c)}</th>"
                       for c in cols)
        body = []
        for r in rows:
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, str):
                    cells.append(f'<td class="text">{escape(v)}</td>')
                elif isinstance(v, int):
                    cells.append(f"<td>{v}</td>")
                else:  # 0-1 score: number over a proportional bar
                    cells.append(f'<td class="bar" style="--v:{v:.3f}" data-v="{v:.4f}"><span>{v:.2f}</span></td>')
            body.append("<tr>" + "".join(cells) + "</tr>")
        return head, "\n".join(body)

    def render(self, metrics: EmbeddingMetrics, fig, title: str, experiment: dict, methods) -> str:
        """experiment: label -> HTML value (dataset, rendering, model, ...); methods: projection keys."""
        n_cls = len(set(metrics.labels.tolist()))
        head, rows = self._class_table(metrics.class_table())
        confused = "\n".join(f'<tr><td class="text">{escape(t)}</td><td class="text">{escape(p)}</td><td>{n}</td></tr>'
                             for t, p, n in metrics.confused_pairs(top=10))
        return self.template.substitute(
            title=escape(title),
            subtitle=f"{len(metrics.labels)} parts, {n_cls} classes. Each part is the average DINOv2 "
                     f"embedding of its rendered views; every part is a query against all the others.",
            experiment=self._facts(experiment),
            methods=self._facts({METHOD_INFO[m][0]: escape(METHOD_INFO[m][1]) for m in methods}),
            cards=self._cards(metrics.summary(), 1 / n_cls),
            figure=self.figure_uri(fig),
            confused_rows=confused or '<tr><td class="text" colspan="3">none</td></tr>',
            class_head=head,
            class_rows=rows,
            generated=f"Generated {datetime.now():%Y-%m-%d %H:%M}",
        )

    def save(self, path, metrics: EmbeddingMetrics, fig, title: str, experiment: dict, methods) -> Path:
        path = Path(path)
        path.write_text(self.render(metrics, fig, title, experiment, methods))
        return path
