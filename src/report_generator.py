"""HTML report generator.

Renders the full analysis report (``src/report_template.html``) with all
results tables and figures embedded as base64 PNG images, producing a single
self-contained ``outputs/report.html`` that can be opened offline, attached
to applications, or hosted (e.g. via GitHub Pages).

The generator is deliberately "dumb": it receives a context dictionary of
ready-made values and HTML fragments, so all analytics remain in the
domain modules and the template stays the only presentation concern.
"""

from __future__ import annotations

import base64
import logging
import pathlib
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

TEMPLATE_DIR = pathlib.Path(__file__).resolve().parent


def _embed(path: Optional[str]) -> str:
    """Base64-embed a PNG image for inline HTML display.

    Parameters
    ----------
    path:
        Path to the PNG file, or ``None``.

    Returns
    -------
    str
        Base64 payload string, or ``""`` when the path is missing.
    """
    if not path or not pathlib.Path(path).exists():
        return ""
    return base64.b64encode(pathlib.Path(path).read_bytes()).decode("ascii")


def _df_to_html(df, max_rows: int = 12, float_fmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a compact styled HTML table.

    Parameters
    ----------
    df:
        Table to render.
    max_rows:
        Maximum number of rows to include.
    float_fmt:
        Format string for float columns.

    Returns
    -------
    str
        HTML table string.
    """
    if df is None or len(df) == 0:
        return "<p><i>no data</i></p>"
    view = df.head(max_rows).copy()
    styled = view.to_html(
        index=False, border=0, classes="tbl", float_format=float_fmt.format,
        escape=True,
    )
    if len(df) > max_rows:
        styled += (
            f"<p style='font-size:12px;color:#5b6b7f;'>"
            f"Showing {max_rows} of {len(df)} rows.</p>"
        )
    return styled


def generate_report(
    context: Dict[str, Any],
    output_path: pathlib.Path,
) -> str:
    """Render the Jinja2 template with the analysis context.

    Parameters
    ----------
    context:
        Values prepared by the pipeline: quality report, EDA summary,
        statistical tests, model tables, backtest stats and figure paths.
    output_path:
        Destination HTML file path.

    Returns
    -------
    str
        Absolute path of the generated report.
    """
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("report_template.html")

    # Embed figures
    fig_paths = context.pop("figures", {})
    context["fig"] = {name: _embed(path) for name, path in fig_paths.items()}

    html = template.render(**context)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    size_mb = output_path.stat().st_size / 1e6
    logger.info(
        "Report written to %s (%.1f MB, %d embedded figures)",
        output_path, size_mb, sum(1 for v in context["fig"].values() if v),
    )
    return str(output_path)
