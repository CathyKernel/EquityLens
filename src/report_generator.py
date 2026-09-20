"""PDF report generator (ReportLab).

Renders the full analysis report as a self-contained, print-ready PDF
(``outputs/report.pdf``): a canvas-drawn cover page, an auto-generated
clickable table of contents, nine numbered sections, embedded EDA/model
figures, formatted result tables and KPI callout strips.

Design notes
------------
* **Portability first.** The generator is part of the repository, so it must
  run on any machine after ``pip install -r requirements.txt``. Fonts are
  resolved from matplotlib's bundled DejaVu family (matplotlib is already a
  hard dependency), with a graceful fallback to PDF core fonts if the TTF
  files cannot be located.
* **Single artefact.** Everything (cover, TOC, body) is produced by one
  ReportLab build - regenerating the report is always a single
  ``python run_pipeline.py --stage report`` away.
* **Separation of concerns.** As with the previous HTML renderer, this module
  is deliberately "dumb": it receives a context dictionary of ready-made
  values and DataFrames from the pipeline; all analytics live in the domain
  modules.

The color system is a fixed, low-saturation cascade palette (XL/L/M/S/XS
tiers) so that large areas stay quiet while small accents carry emphasis.
"""

from __future__ import annotations

import hashlib
import logging
import math
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page geometry (A4 portrait)
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = A4
MARGIN_L = 52.0
MARGIN_R = 52.0
MARGIN_T = 66.0   # room for the running header
MARGIN_B = 58.0   # room for the running footer
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R          # ~491pt
MAX_IMG_H = 286.0                                  # keeps figure + caption on one page

# ---------------------------------------------------------------------------
# Cascade palette (design_engine.py palette-cascade, seed 42, mode minimal)
# Fixed design system for the report - do not hand-edit the hex values.
# ---------------------------------------------------------------------------
PAGE_BG      = colors.HexColor("#f4f5f5")   # XL tier: backgrounds
SECTION_BG   = colors.HexColor("#f0f1f2")
CARD_BG      = colors.HexColor("#e8eaeb")   # L tier: surfaces
TABLE_STRIPE = colors.HexColor("#ebeded")
HEADER_FILL  = colors.HexColor("#32454e")   # M tier: structural fills
COVER_BLOCK  = colors.HexColor("#566a74")
BORDER       = colors.HexColor("#acbdc5")   # S tier: edges
ICON         = colors.HexColor("#4b86a4")
ACCENT       = colors.HexColor("#1f6c92")   # XS tier: emphasis
ACCENT_2     = colors.HexColor("#c23a50")
TEXT_PRIMARY = colors.HexColor("#131515")
TEXT_MUTED   = colors.HexColor("#747b7e")
SEM_SUCCESS  = colors.HexColor("#529067")
SEM_ERROR    = colors.HexColor("#a25b54")

REPORT_TITLE = "EquityLens"
REPORT_SUBTITLE = (
    "Statistical Analysis and Machine-Learning Trading Signals "
    "on the S&P 500 Universe"
)
# XML entities are assembled at runtime (rather than written as literals) so
# that source-level HTML-unescaping tools cannot collapse them into raw
# characters, which would corrupt Paragraph markup parsing.
_AMP = "&" + "amp;"
_LT = "&" + "lt;"
_GT = "&" + "gt;"
# Paragraph-safe title fragments ("&" must be escaped inside Paragraph XML)
TITLE_P = "EquityLens"
SPX_P = "S" + _AMP + "P 500"

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def _setup_fonts() -> Dict[str, str]:
    """Register the DejaVu font family from matplotlib's bundled TTF files.

    matplotlib ships the full DejaVu family inside every wheel/pip install,
    which makes it the most portable high-quality font source for this
    project. If the files cannot be found (exotic environments), the
    generator falls back to the PDF core fonts so the pipeline never breaks.

    Returns
    -------
    dict
        Mapping of logical role ("sans", "sans_bold", "serif", "serif_bold")
        to the registered ReportLab font name.
    """
    try:
        font_dir = pathlib.Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    except Exception:  # pragma: no cover - matplotlib always exposes this
        font_dir = None

    spec = {
        "R-Sans": "DejaVuSans.ttf",
        "R-Sans-Bold": "DejaVuSans-Bold.ttf",
        "R-Sans-Italic": "DejaVuSans-Oblique.ttf",
        "R-Serif": "DejaVuSerif.ttf",
        "R-Serif-Bold": "DejaVuSerif-Bold.ttf",
        "R-Serif-Italic": "DejaVuSerif-Italic.ttf",
    }
    registered: Dict[str, str] = {}
    if font_dir is not None and font_dir.is_dir():
        for name, fname in spec.items():
            path = font_dir / fname
            if path.exists():
                try:
                    pdfmetrics.registerFont(TTFont(name, str(path)))
                    registered[name] = name
                except Exception:  # pragma: no cover
                    pass

    if "R-Serif" not in registered:
        # Portability escape hatch: PDF core fonts (present in every viewer).
        registered = {
            "R-Sans": "Helvetica",
            "R-Sans-Bold": "Helvetica-Bold",
            "R-Sans-Italic": "Helvetica-Oblique",
            "R-Serif": "Times-Roman",
            "R-Serif-Bold": "Times-Bold",
            "R-Serif-Italic": "Times-Italic",
        }
        logger.warning(
            "DejaVu TTF fonts not found; falling back to PDF core fonts."
        )

    registerFontFamily(
        "R-Serif",
        normal=registered["R-Serif"],
        bold=registered["R-Serif-Bold"],
        italic=registered["R-Serif-Italic"],
        boldItalic=registered["R-Serif-Bold"],
    )
    registerFontFamily(
        "R-Sans",
        normal=registered["R-Sans"],
        bold=registered["R-Sans-Bold"],
        italic=registered["R-Sans-Italic"],
        boldItalic=registered["R-Sans-Bold"],
    )
    return {
        "sans": registered["R-Sans"],
        "sans_bold": registered["R-Sans-Bold"],
        "sans_italic": registered["R-Sans-Italic"],
        "serif": registered["R-Serif"],
        "serif_bold": registered["R-Serif-Bold"],
        "serif_italic": registered["R-Serif-Italic"],
    }


# ---------------------------------------------------------------------------
# Document template with auto-TOC support
# ---------------------------------------------------------------------------
class TocDocTemplate(SimpleDocTemplate):
    """SimpleDocTemplate that feeds heading flowables into the TOC.

    Every heading Paragraph carrying a ``bookmark_key`` attribute triggers a
    ``TOCEntry`` notification, which the :class:`TableOfContents` flowable
    collects across ``multiBuild`` passes until page numbers stabilise.
    """

    def afterFlowable(self, flowable):
        if hasattr(flowable, "bookmark_name"):
            level = getattr(flowable, "bookmark_level", 0)
            text = getattr(flowable, "bookmark_text", "")
            key = getattr(flowable, "bookmark_key", "")
            self.notify("TOCEntry", (level, text, self.page, key))


# ---------------------------------------------------------------------------
# Page furniture: canvas-drawn cover, running header and footer
# ---------------------------------------------------------------------------
def _make_cover_painter(ctx: Dict[str, Any], fonts: Dict[str, str]):
    """Build the onFirstPage callback that paints the report cover."""

    def paint(canvas, doc) -> None:
        c = canvas
        c.saveState()
        # Full-bleed quiet background
        c.setFillColor(PAGE_BG)
        c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

        # Restrained structural block, top-right (auto-sized to its label so
        # the text can never poke out of the block on the left)
        cover_tag = "END-TO-END RESEARCH PIPELINE"
        c.setFont(fonts["sans_bold"], 8.5)
        tag_w = pdfmetrics.stringWidth(cover_tag, fonts["sans_bold"], 8.5)
        block_w = tag_w + 24.0
        c.setFillColor(COVER_BLOCK)
        c.rect(PAGE_W - MARGIN_R - block_w, PAGE_H - 86, block_w, 26, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawRightString(PAGE_W - MARGIN_R - 12, PAGE_H - 77, cover_tag)

        # Kicker
        c.setFillColor(TEXT_MUTED)
        c.setFont(fonts["sans"], 10)
        c.drawString(MARGIN_L, PAGE_H - 130, "QUANTITATIVE RESEARCH REPORT")
        c.setStrokeColor(ACCENT)
        c.setLineWidth(2.2)
        c.line(MARGIN_L, PAGE_H - 140, MARGIN_L + 148, PAGE_H - 140)

        # Title block (chunked per cover typography rules)
        c.setFillColor(TEXT_PRIMARY)
        c.setFont(fonts["sans_bold"], 38)
        c.drawString(MARGIN_L, PAGE_H - 200, "EquityLens")
        c.setFillColor(COVER_BLOCK)
        c.setFont(fonts["serif"], 16.5)
        c.drawString(
            MARGIN_L, PAGE_H - 232,
            "Statistical Analysis and Machine-Learning Trading Signals",
        )
        c.drawString(MARGIN_L, PAGE_H - 254, "on the S&P 500 Universe")

        # Accent bar anchoring the title block
        c.setFillColor(ACCENT)
        c.rect(MARGIN_L - 18, PAGE_H - 258, 4, 74, stroke=0, fill=1)

        # KPI strip card
        win = ctx["window"]
        uni = ctx["universe"]
        stats = [
            (f"{uni['size']}", "S&P 500 constituents"),
            (f"{uni['sectors']}", "GICS sectors"),
            (f"{win['sessions']:,}", "trading sessions"),
            (f"{ctx['quality']['rows_output']:,}", "daily observations"),
        ]
        card_x, card_y, card_w, card_h = (
            MARGIN_L, PAGE_H - 400, CONTENT_W, 74,
        )
        c.setFillColor(CARD_BG)
        c.setStrokeColor(BORDER)
        c.setLineWidth(0.8)
        c.roundRect(card_x, card_y, card_w, card_h, 6, stroke=1, fill=1)
        col_w = card_w / len(stats)
        for i, (value, label) in enumerate(stats):
            cx = card_x + col_w * (i + 0.5)
            c.setFillColor(ACCENT)
            c.setFont(fonts["sans_bold"], 15.5)
            c.drawCentredString(cx, card_y + 40, value)
            c.setFillColor(TEXT_MUTED)
            c.setFont(fonts["sans"], 8)
            c.drawCentredString(cx, card_y + 22, label)
            if i > 0:
                c.setStrokeColor(BORDER)
                c.setLineWidth(0.6)
                c.line(card_x + col_w * i, card_y + 12,
                       card_x + col_w * i, card_y + card_h - 12)

        # Methodology summary, two measured centred lines (never overflow)
        c.setFillColor(TEXT_MUTED)
        c.setFont(fonts["sans"], 9)
        c.drawCentredString(
            PAGE_W / 2, card_y - 30,
            "Data acquisition  -  SQLite warehouse  -  statistical testing",
        )
        c.drawCentredString(
            PAGE_W / 2, card_y - 46,
            "leakage-safe features  -  ML ensemble  -  net-of-cost backtest",
        )

        # Meta block, bottom
        c.setStrokeColor(BORDER)
        c.setLineWidth(0.8)
        c.line(MARGIN_L, 148, PAGE_W - MARGIN_R, 148)
        c.setFillColor(TEXT_PRIMARY)
        c.setFont(fonts["sans"], 9.5)
        c.drawString(MARGIN_L, 126, f"Sample window:  {win['start']}  to  {win['end']}")
        c.drawString(MARGIN_L, 110, f"Benchmark:  {uni['benchmark']}  (SPDR S&P 500 ETF Trust)")
        c.setFillColor(TEXT_MUTED)
        c.drawString(MARGIN_L, 94, f"Generated:  {ctx['generated_at']}")
        c.setFont(fonts["sans"], 8.5)
        c.drawString(
            MARGIN_L, 70,
            "Python - pandas - SQLite - scikit-learn - XGBoost - statsmodels - ReportLab",
        )
        c.restoreState()

    return paint


def _make_furniture_painter(fonts: Dict[str, str], generated_at: str):
    """Build the onLaterPages callback: running header + page footer."""

    def paint(canvas, doc) -> None:
        c = canvas
        c.saveState()
        # Header: muted title (left) + generation date (right) + accent rule
        c.setFillColor(TEXT_MUTED)
        c.setFont(fonts["sans"], 7.5)
        c.drawString(MARGIN_L, PAGE_H - 40, "EquityLens - S&P 500 Research Report")
        c.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 40, f"Generated {generated_at}")
        c.setStrokeColor(ACCENT)
        c.setLineWidth(1.4)
        c.line(MARGIN_L, PAGE_H - 48, PAGE_W - MARGIN_R, PAGE_H - 48)
        # Footer: light rule + project label left + page number right
        c.setStrokeColor(BORDER)
        c.setLineWidth(0.6)
        c.line(MARGIN_L, 44, PAGE_W - MARGIN_R, 44)
        c.setFillColor(TEXT_MUTED)
        c.setFont(fonts["sans"], 7.5)
        c.drawString(MARGIN_L, 32, "equitylens")
        c.drawRightString(PAGE_W - MARGIN_R, 32, f"Page {c.getPageNumber()}")
        c.restoreState()

    return paint

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def esc(value: Any) -> str:
    """Escape a value for safe embedding inside Paragraph XML markup."""
    return (
        str(value)
        .replace("&", _AMP)
        .replace("<", _LT)
        .replace(">", _GT)
    )


def fmt_p(p: Optional[float]) -> str:
    """Format a p-value, using scientific notation with a superscript.

    Values below 1e-4 are rendered as ``m × 10<super>e</super>`` inside the
    Paragraph markup (never raw unicode superscripts).
    """
    if p is None:
        return "n/a"
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n/a"
    if p != 0 and p < 1e-300:
        return _LT + " 10<super>-300</super>"
    if p == 0.0:
        return _LT + " 10<super>-300</super>"
    if p < 1e-4:
        exp = math.floor(math.log10(p))
        mant = p / (10 ** exp)
        return f"{mant:.1f} × 10<super>{exp}</super>"
    return f"{p:.3f}"


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def _build_styles(fonts: Dict[str, str]) -> Dict[str, ParagraphStyle]:
    """Create the paragraph style sheet for the report body."""
    sans, sans_b = fonts["sans"], fonts["sans_bold"]
    serif = fonts["serif"]
    return {
        "body": ParagraphStyle(
            "body", fontName=serif, fontSize=10, leading=15.5,
            alignment=TA_JUSTIFY, textColor=TEXT_PRIMARY,
            spaceBefore=0, spaceAfter=8,
        ),
        "lead": ParagraphStyle(
            "lead", fontName=serif, fontSize=10.5, leading=16.5,
            alignment=TA_JUSTIFY, textColor=TEXT_PRIMARY, spaceAfter=10,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName=serif, fontSize=10, leading=15,
            alignment=TA_LEFT, textColor=TEXT_PRIMARY,
            leftIndent=16, bulletIndent=5, spaceBefore=0, spaceAfter=5,
        ),
        "h1": ParagraphStyle(
            "h1", fontName=sans_b, fontSize=16.5, leading=21,
            textColor=HEADER_FILL, spaceBefore=20, spaceAfter=9,
        ),
        "h2": ParagraphStyle(
            "h2", fontName=sans_b, fontSize=12.5, leading=16.5,
            textColor=TEXT_PRIMARY, spaceBefore=14, spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "caption", fontName=sans, fontSize=8.5, leading=11.5,
            alignment=TA_CENTER, textColor=TEXT_MUTED,
            spaceBefore=5, spaceAfter=4,
        ),
        "tbl_header": ParagraphStyle(
            "tbl_header", fontName=sans, fontSize=8.6, leading=11,
            alignment=TA_CENTER, textColor=colors.white,
        ),
        "tbl_cell": ParagraphStyle(
            "tbl_cell", fontName=sans, fontSize=8.4, leading=11,
            alignment=TA_CENTER, textColor=TEXT_PRIMARY,
        ),
        "tbl_cell_l": ParagraphStyle(
            "tbl_cell_l", fontName=sans, fontSize=8.4, leading=11,
            alignment=TA_LEFT, textColor=TEXT_PRIMARY,
        ),
        "tbl_cell_r": ParagraphStyle(
            "tbl_cell_r", fontName=sans, fontSize=8.4, leading=11,
            alignment=TA_RIGHT, textColor=TEXT_PRIMARY,
        ),
        "stat_big": ParagraphStyle(
            "stat_big", fontName=sans, fontSize=13.5, leading=16,
            alignment=TA_CENTER, textColor=ACCENT,
        ),
        "stat_label": ParagraphStyle(
            "stat_label", fontName=sans, fontSize=7.3, leading=9.5,
            alignment=TA_CENTER, textColor=TEXT_MUTED,
        ),
        "toc_title": ParagraphStyle(
            "toc_title", fontName=sans_b, fontSize=16.5, leading=21,
            textColor=HEADER_FILL, spaceBefore=6, spaceAfter=14,
        ),
        "toc0": ParagraphStyle(
            "toc0", fontName=sans_b, fontSize=10.5, leading=17,
            textColor=TEXT_PRIMARY, leftIndent=6,
        ),
        "toc1": ParagraphStyle(
            "toc1", fontName=serif, fontSize=9.8, leading=15,
            textColor=TEXT_PRIMARY, leftIndent=24,
        ),
    }


def _heading(text: str, style: ParagraphStyle, level: int = 0) -> Paragraph:
    """Create a bookmarked heading that feeds the auto-generated TOC.

    The heading text is expected to be markup-safe (caller escapes "&").
    """
    key = "h_" + hashlib.md5(text.encode()).hexdigest()[:8]
    p = Paragraph(f'<a name="{key}"/><b>{text}</b>', style)
    p.bookmark_name = key
    p.bookmark_level = level
    p.bookmark_text = text.replace(_AMP, "&")
    p.bookmark_key = key
    return p


# ---------------------------------------------------------------------------
# Table and figure helpers
# ---------------------------------------------------------------------------
_ALIGN_STYLE = {"L": "tbl_cell_l", "C": "tbl_cell", "R": "tbl_cell_r"}


def _make_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    styles: Dict[str, ParagraphStyle],
    ratios: Optional[Sequence[float]] = None,
    aligns: Optional[Sequence[str]] = None,
    repeat_header: bool = False,
) -> Table:
    """Build a centred, zebra-striped table from markup-ready cell strings.

    Parameters
    ----------
    headers:
        Plain-text column headers (auto-escaped and bolded).
    rows:
        Row content; each cell must already be Paragraph-markup-safe
        (use :func:`esc` on raw text, :func:`fmt_p` for p-values).
    ratios:
        Column width proportions (must sum to ~1.0); defaults to equal.
    aligns:
        Per-column alignment, one of "L" / "C" / "R" (default "C").
    repeat_header:
        Repeat the header row when the table breaks across pages.
    """
    n_cols = len(headers)
    if ratios is None:
        ratios = [1.0 / n_cols] * n_cols
    total = sum(ratios)
    col_widths = [r / total * CONTENT_W for r in ratios]
    if aligns is None:
        aligns = ["C"] * n_cols

    data = [
        [Paragraph(f"<b>{esc(h)}</b>", styles["tbl_header"]) for h in headers]
    ]
    for row in rows:
        data.append([
            Paragraph(cell, styles[_ALIGN_STYLE[a]])
            for cell, a in zip(row, aligns)
        ])

    table = Table(
        data, colWidths=col_widths, hAlign="CENTER",
        repeatRows=1 if repeat_header else 0,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_FILL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TABLE_STRIPE]),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]))
    return table


def _embed_image(path: str, max_width: float = CONTENT_W,
                 max_height: float = MAX_IMG_H) -> Image:
    """Embed a PNG with its aspect ratio preserved and size constrained."""
    reader = ImageReader(path)
    orig_w, orig_h = reader.getSize()
    ratio = min(
        max_width / orig_w if orig_w > max_width else 1.0,
        max_height / orig_h if orig_h > max_height else 1.0,
    )
    img = Image(path, width=orig_w * ratio, height=orig_h * ratio)
    img.hAlign = "CENTER"
    return img


def _figure_block(
    path: Optional[str], caption: str, styles: Dict[str, ParagraphStyle],
) -> List[Any]:
    """Return the flowables for a figure (image + caption, kept together)."""
    if not path or not pathlib.Path(path).exists():
        logger.warning("Figure not found, skipping: %s", path)
        return []
    return [KeepTogether([
        _embed_image(path),
        Paragraph(caption, styles["caption"]),
    ])]


def _stat_row(
    items: Sequence[Tuple[str, str]],
    styles: Dict[str, ParagraphStyle],
) -> Table:
    """Build a row of KPI callout boxes (value on top, label below)."""
    n = len(items)
    gap = 8.0
    box_w = (CONTENT_W - gap * (n - 1)) / n
    cells: List[Any] = []
    widths: List[float] = []
    for i, (value, label) in enumerate(items):
        inner = Table(
            [
                [Paragraph(f"<b>{esc(value)}</b>", styles["stat_big"])],
                [Paragraph(esc(label), styles["stat_label"])],
            ],
            colWidths=[box_w],
        )
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
            ("TOPPADDING", (0, 1), (-1, 1), 1),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
        cells.append(inner)
        widths.append(box_w)
        if i < n - 1:
            cells.append("")
            widths.append(gap)
    outer = Table([cells], colWidths=widths, hAlign="CENTER")
    outer.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return outer


def _bullets(items: Sequence[str], styles: Dict[str, ParagraphStyle]) -> List[Paragraph]:
    """Build a left-aligned bullet list (never justified, per list rules)."""
    return [Paragraph(item, styles["bullet"], bulletText="•") for item in items]


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def generate_report(
    context: Dict[str, Any],
    output_path: pathlib.Path,
) -> str:
    """Render the full PDF analysis report from the pipeline context.

    Parameters
    ----------
    context:
        Values prepared by the pipeline: window and universe descriptors,
        data-quality report, SQL aggregates, EDA summary, statistical tests,
        model tables, backtest stats, parameters and figure paths.
    output_path:
        Destination PDF file path.

    Returns
    -------
    str
        Absolute path of the generated report.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fonts = _setup_fonts()
    S = _build_styles(fonts)
    q = context["quality"]
    win = context["window"]
    uni = context["universe"]
    eda = context["eda"]
    stats = context["stats"]
    ds = context["dataset"]
    mod = context["modeling"]
    bt = context["backtest"]
    par = context["params"]
    figs = context.get("figures", {})

    best_name = mod["best_model"]
    best = mod["test_metrics"][best_name]
    strategy, benchmark = bt["strategy"], bt["benchmark"]
    cv_df = mod["cv"]

    story: List[Any] = []
    fig_n = 0
    tbl_n = 0

    def P(text: str, style: str = "body") -> None:
        story.append(Paragraph(text, S[style]))

    def H1(number: int, title: str) -> None:
        story.append(CondPageBreak(130))
        story.append(_heading(f"{number}. {esc(title)}", S["h1"], level=0))

    def H2(number: str, title: str) -> None:
        story.append(_heading(f"{number} {esc(title)}", S["h2"], level=1))

    def BUL(items: Sequence[str]) -> None:
        story.extend(_bullets(items, S))

    def FIG(name: str, caption: str) -> None:
        nonlocal fig_n
        fig_n += 1
        story.extend(_figure_block(figs.get(name), f"Figure {fig_n}. {caption}", S))
        story.append(Spacer(1, 8))

    def TBL(headers: Sequence[str], rows: Sequence[Sequence[str]],
            caption: str, ratios=None, aligns=None) -> None:
        nonlocal tbl_n
        tbl_n += 1
        story.append(Spacer(1, 10))
        story.append(_make_table(headers, rows, S, ratios, aligns))
        story.append(Paragraph(f"Table {tbl_n}. {caption}", S["caption"]))
        story.append(Spacer(1, 12))

    def STATS(items: Sequence[Tuple[str, str]]) -> None:
        story.append(Spacer(1, 4))
        story.append(_stat_row(items, S))
        story.append(Spacer(1, 9))

    # ---- Cover (page 1, painted by onFirstPage) ---------------------------
    story.append(PageBreak())

    # ---- Table of contents --------------------------------------------------
    story.append(Paragraph("<b>Contents</b>", S["toc_title"]))
    toc = TableOfContents()
    toc.levelStyles = [S["toc0"], S["toc1"]]
    toc.dotsMinLevel = 0
    story.append(toc)
    story.append(PageBreak())

    # =========================================================================
    # 1. Executive summary
    # =========================================================================
    H1(1, "Executive Summary")
    P(
        f"This report documents an end-to-end quantitative research pipeline built on "
        f"{uni['size']} large-cap {SPX_P} constituents drawn from all {uni['sectors']} GICS "
        f"sectors, covering {win['sessions']:,} trading sessions from {win['start']} to "
        f"{win['end']} ({q['rows_output']:,} cleaned daily observations). The project walks "
        f"the full research stack: automated data acquisition and validation, a SQLite "
        f"analytical warehouse with window-function SQL, exploratory data analysis, a formal "
        f"hypothesis-testing suite, leakage-safe feature engineering, and a three-model "
        f"machine-learning comparison evaluated with time-series cross-validation and a "
        f"chronological hold-out set. The research question is deliberately modest and "
        f"economically meaningful: can technical and volume features predict the direction "
        f"of the next trading week well enough to improve risk-adjusted returns over a "
        f"passive benchmark, net of transaction costs?"
    )
    STATS([
        (f"{best['roc_auc']:.3f}", f"hold-out ROC-AUC ({best_name})"),
        (f"{best['accuracy']:.1%}", "hold-out accuracy"),
        (f"{strategy['sharpe']:.2f}", "strategy Sharpe (OOS)"),
        (f"{benchmark['sharpe']:.2f}", f"{uni['benchmark']} Sharpe (OOS)"),
    ])
    STATS([
        (f"{strategy['max_drawdown_pct']:.1f}%", "strategy max drawdown"),
        (f"{benchmark['max_drawdown_pct']:.1f}%", f"{uni['benchmark']} max drawdown"),
        (f"{ds['rows']:,}", "model rows"),
        (f"{ds['n_features']}", "features"),
    ])
    P("Key findings, each quantified in the sections that follow:", "body")
    BUL([
        f"<b>Non-Gaussian returns.</b> Jarque-Bera rejects normality for "
        f"{stats['normality']['n_reject']} of {stats['normality']['n_tickers']} tickers "
        f"(pooled excess kurtosis {stats['normality']['pooled']['excess_kurtosis']:.1f}), "
        f"confirming fat tails that Gaussian risk metrics would understate.",
        f"<b>Stationarity discipline.</b> Augmented Dickey-Fuller tests reject a unit root "
        f"in daily returns but not in prices, formally justifying return-based modelling.",
        f"<b>Volatility clustering.</b> Ljung-Box tests on squared returns are highly "
        f"significant (ARCH effects), motivating the rolling-volatility and Bollinger-band "
        f"feature families that dominate the importance ranking.",
        f"<b>Sector heterogeneity.</b> One-way ANOVA across sectors yields "
        f"F = {stats['sector_anova']['f_stat']:.2f}, "
        f"p = {stats['sector_anova']['p_value']:.3f}, informing the one-hot sector encoding.",
        f"<b>Model ranking.</b> {best_name} wins the hold-out comparison with ROC-AUC "
        f"{best['roc_auc']:.3f} and accuracy {best['accuracy']:.1%} against a "
        f"{best['majority_class_share']:.1%} up-period base rate - a modest but consistent "
        f"edge, in line with weak-form market efficiency.",
        f"<b>Economic significance.</b> The {par['horizon']}-day rebalanced, "
        f"probability-threshold strategy (P(up) ≥ {par['threshold']:.2f}) returns an "
        f"out-of-sample Sharpe of {strategy['sharpe']:.2f} versus {benchmark['sharpe']:.2f} "
        f"for {uni['benchmark']} buy-and-hold over identical windows, with maximum drawdown "
        f"{strategy['max_drawdown_pct']:.1f}% versus {benchmark['max_drawdown_pct']:.1f}%, "
        f"net of {par['cost_bps']} bps turnover costs.",
    ])

    # =========================================================================
    # 2. Data pipeline and governance
    # =========================================================================
    H1(2, "Data Pipeline and Governance")
    H2("2.1", "Acquisition and Storage")
    P(
        f"Daily bars for the {uni['size']}-stock universe plus the {uni['benchmark']} "
        f"benchmark are pulled from Yahoo Finance via <i>yfinance</i>, using "
        f"dividend-and-split-adjusted closes so that returns are total-return comparable "
        f"across tickers. Every symbol is cached as an individual Parquet file, which makes "
        f"re-runs deterministic, offline-friendly and cheap: a full refresh touches the "
        f"network once, while subsequent runs - including the unit-test suite - read from "
        f"the local columnar cache. The raw panel is then loaded into a SQLite warehouse "
        f"({q['rows_output']:,} rows) with a composite primary key on "
        f"<i>(ticker, date)</i>, an index on <i>date</i>, and a companion "
        f"<i>ticker_sector</i> mapping table that powers the sector-level SQL analytics."
    )
    H2("2.2", "Universe Construction")
    P(
        f"The universe intentionally spans all eleven GICS sectors rather than a handful "
        f"of mega-cap technology names. Sector completeness matters for three reasons: "
        f"the sector ANOVA in Section 5 needs a balanced cross-section, the correlation "
        f"structure in Section 4 is far more informative across industries than within "
        f"one, and the machine-learning panel gains cross-sectional variation that a "
        f"single-sector universe would lack. Each sector contributes between three and "
        f"eleven constituents, weighted toward the larger, more liquid GICS groups."
    )
    sec_rows = [[esc(s), str(n)] for s, n in context["sector_counts"]]
    sec_rows.append(["<b>Total</b>", f"<b>{uni['size']}</b>"])
    TBL(
        ["GICS sector", "Constituents"], sec_rows,
        "Universe composition by GICS sector.",
        ratios=[0.7, 0.3], aligns=["L", "C"],
    )
    H2("2.3", "Cleaning and Validation")
    P(
        "The cleaning layer applies production-grade hygiene rules in a fixed order: "
        "de-duplication on (ticker, date), OHLC integrity checks "
        "(high must dominate open/close, low must undercut them), calendar alignment of "
        "every ticker onto the union trading calendar so the panel is strictly "
        "rectangular, and forward-filling of isolated gaps caused by trading halts. "
        "Rows that remain missing after forward-fill correspond to pre-listing dates and "
        "are dropped. Extreme daily moves beyond five standard deviations are flagged "
        "but deliberately kept: in equity markets, crashes and earnings shocks are "
        "genuine information, and trimming them would bias every risk estimate in the "
        "study. The full audit trail is summarised below."
    )
    q_rows = [
        ["Rows ingested", f"{q['rows_input']:,}", "yfinance download, universe + benchmark"],
        ["Duplicate rows removed", str(q["duplicates_removed"]), "keep last on (ticker, date)"],
        ["OHLC integrity violations", str(q["ohlc_violations"]), "high/low vs open/close"],
        ["Invalid closes", str(q["invalid_closes"]), "non-positive or missing"],
        ["Gaps forward-filled", str(q["gaps_forward_filled"]), "trading-halt policy"],
        ["Unfillable rows dropped", str(q["gaps_unfillable_dropped"]), "pre-listing dates"],
        ["Outlier days flagged (kept)", str(q["outlier_days_flagged"]),
         f"beyond {q['outlier_threshold_sigma']:.0f} sigma, retained"],
        ["Tickers kept", str(q["tickers_kept"]), f"minimum {260} sessions of history"],
        ["Trading sessions", f"{q['trading_days']:,}", "union calendar, rectangular panel"],
        ["Rows output", f"{q['rows_output']:,}", "cleaned analysis panel"],
    ]
    TBL(
        ["Quality metric", "Value", "Rule / detail"], q_rows,
        "Data-quality audit of the cleaning stage.",
        ratios=[0.34, 0.2, 0.46], aligns=["L", "R", "L"],
    )

    # =========================================================================
    # 3. SQL analytical layer
    # =========================================================================
    H1(3, "SQL Analytical Layer")
    P(
        "In production quant stacks, analysts rarely touch raw files: market data lives "
        "in a warehouse and is extracted through SQL. This project mirrors that workflow. "
        "The SQLite database is queried through a catalogue of five analytical statements "
        "that exercise the techniques data-science interviews probe for: common table "
        "expressions, window functions (<i>LAG</i> over partitions, <i>ROW_NUMBER</i> for "
        "period-end snapshots, rolling <i>AVG ... ROWS BETWEEN 29 PRECEDING AND CURRENT "
        "ROW</i> for 30-day volatility), grouped aggregation and JOINs against the sector "
        "mapping. Three of the headline extracts are reproduced below."
    )
    sql = context["sql"]
    sec_agg = sql["sector_aggregate"]
    sa_rows = [
        [esc(r["sector"]), f"{int(r['n_obs']):,}", f"{r['mean_daily_ret_pct']:.4f}",
         f"{r['ann_vol_pct']:.1f}", f"{r['annualised_sharpe']:.2f}"]
        for _, r in sec_agg.iterrows()
    ]
    TBL(
        ["Sector", "Obs", "Mean daily ret (%)", "Ann. vol (%)", "Ann. Sharpe"],
        sa_rows,
        "Sector aggregates computed in pure SQL (window functions + grouped aggregation, "
        "annualised Sharpe with a 2% risk-free rate).",
        ratios=[0.3, 0.16, 0.22, 0.16, 0.16], aligns=["L", "R", "R", "R", "R"],
    )
    ann = sql["annual_summary"]
    ann_rows = [
        [esc(str(int(float(r["year"])))), f"{r['mean_annual_return_pct']:+.1f}",
         f"{r['share_positive_pct']:.0f}"]
        for _, r in ann.iterrows()
    ]
    TBL(
        ["Year", "Mean annual return (%)", "Tickers positive (%)"],
        ann_rows,
        "Universe-average annual returns by calendar year (SQL annual summary, "
        "year-end closes via ROW_NUMBER).",
        ratios=[0.24, 0.4, 0.36], aligns=["C", "R", "R"],
    )
    league = sql["volatility_league"]
    vl_rows = [
        [esc(r["ticker"]), esc(r["sector"]), f"{r['annual_vol_pct']:.1f}",
         f"{r['annualised_sharpe']:.2f}"]
        for _, r in league.iterrows()
    ]
    TBL(
        ["Ticker", "Sector", "Ann. vol (%)", "Ann. Sharpe"],
        vl_rows,
        "Ten most volatile constituents (SQL volatility league).",
        ratios=[0.2, 0.4, 0.2, 0.2], aligns=["C", "L", "R", "R"],
    )
    P(
        "The extracts already sketch the investment story: realised volatility and "
        "risk-adjusted performance differ widely across sectors, and single years swing "
        "the cross-sectional mean by double digits - a first hint of the regime "
        "dependence that the statistical tests in Section 5 formalise."
    )

    # =========================================================================
    # 4. Exploratory data analysis
    # =========================================================================
    H1(4, "Exploratory Data Analysis")
    P(
        f"All EDA statistics are computed on the cleaned panel so that every number in "
        f"this section matches the modelling dataset exactly. The pooled daily return "
        f"has mean {eda['pooled_mean_daily_pct']:.4f} and standard deviation "
        f"{eda['pooled_std_daily_pct']:.4f}, with the worst single day at "
        f"{eda['pooled_min_pct']:.1f}% and the best at {eda['pooled_max_pct']:.1f}%. "
        f"The best-performing constituent, {eda['best_ticker_total']}, multiplied "
        f"{eda['best_ticker_total_pct']:.0f}% over the window, and the average pairwise "
        f"return correlation across the universe is {eda['avg_cross_correlation']:.2f} - "
        f"diversification is real but incomplete, a fact the strategy exploits by staying "
        f"selective."
    )
    FIG("price_evolution",
        "Normalised price evolution of the universe (base 100, log scale). The black "
        "line is the equal-weight universe index; grey lines are individual constituents.")
    P(
        "The log-scale view highlights both the dispersion of outcomes - order-of-"
        "magnitude differences in cumulative return among survivors of the same index - "
        "and the shared regime structure: the 2020 drawdown and the 2022-2023 pattern "
        "hit every line at the same dates. Common-factor risk dominates idiosyncratic "
        "risk in stressed markets, which is precisely why the correlation heatmap below "
        "is central to the risk story."
    )
    FIG("return_distribution",
        "Pooled daily return distribution versus a Gaussian with identical mean and "
        "variance (log-density scale).")
    P(
        "The histogram makes the fat-tail evidence visible long before any formal test: "
        "the empirical density is taller at the centre and dramatically heavier beyond "
        "three standard deviations than the matched Gaussian. The practical consequence "
        "for this project is twofold - risk metrics based on normality would understate "
        "tail risk, and models that impose Gaussian structure on returns are "
        "misspecified by construction."
    )
    FIG("volatility_league",
        "Annualised volatility by GICS sector: bars show the sector mean, whiskers the "
        "most and least volatile constituents in the sector, and the dashed line the "
        "universe-median ticker.")
    P(
        "Volatility ranking is strongly sector-structured: cyclical growth sectors "
        "(Consumer Discretionary, Technology, Communication Services) cluster at the "
        "top, defensive groups (Utilities, Consumer Staples, Health Care) at the bottom. "
        "Within-sector whisker ranges show that idiosyncratic names like Tesla can "
        "out-volatilise their sector mean by a wide margin, so sector dummies alone "
        "cannot capture risk - the per-ticker rolling volatility features in Section 6 "
        "are necessary."
    )
    FIG("correlation_heatmap",
        "Average pairwise correlation of daily returns within and across GICS sectors "
        "(diagonal = within-sector average, self-correlations excluded).")
    P(
        "The block structure of the correlation matrix is the diversification argument "
        "in one picture: sectors such as Utilities, Consumer Staples and Real Estate "
        "correlate weakly with the Technology block, while intra-sector correlations "
        "run visibly higher than cross-sector ones. An equal-weight portfolio over this "
        "universe therefore earns a genuine - though bounded - diversification benefit, "
        "and the long-only strategy can reduce variance further by being picky about "
        "when to be invested at all."
    )
    FIG("sector_risk_return",
        "Sector risk/return profile: equal-weight annualised mean return versus "
        "annualised volatility per GICS sector, full sample.")
    FIG("sector_boxplot",
        "Distribution of daily returns by sector (boxes span the interquartile range, "
        "whiskers 1.5x IQR, outliers hidden for readability).")
    P(
        "Risk and return are broadly rank-correlated across sectors, as standard theory "
        "predicts, with the boxplot confirming that the ranking is driven by dispersion "
        "rather than by mean shifts alone. Together, these exhibits justify the feature "
        "families chosen for the model: trend and momentum for the return structure, "
        "rolling volatility for the risk structure, and sector one-hots for the "
        "cross-sectional grouping."
    )

    # =========================================================================
    # 5. Statistical hypothesis tests
    # =========================================================================
    H1(5, "Statistical Hypothesis Tests")
    P(
        "Before any model is fitted, the statistical properties of the return series "
        "are validated formally - this is what separates a defensible quant workflow "
        "from throwing scikit-learn at a CSV. Five families of tests are applied, each "
        "chosen to license a specific modelling decision made later in the pipeline."
    )
    norm, stat_adf, autoc, anova = (
        stats["normality"], stats["stationarity"],
        stats["autocorrelation"], stats["sector_anova"],
    )
    test_rows = [
        ["Jarque-Bera, pooled", f"{norm['pooled']['jb_stat']:,.0f}",
         fmt_p(norm["pooled"]["p_value"]), "Normality rejected"],
        ["Jarque-Bera, per ticker",
         f"{norm['n_reject']} / {norm['n_tickers']} rejected", "-",
         "Fat tails across the universe"],
        ["ADF, price-like series", f"{stat_adf['price_adf_stat']:.2f}",
         fmt_p(stat_adf["price_p_value"]), "Unit root not rejected"],
        ["ADF, daily returns", f"{stat_adf['returns_adf_stat']:.2f}",
         fmt_p(stat_adf["returns_p_value"]), "Stationary"],
        ["Ljung-Box, returns (lag 10)",
         f"{autoc['returns_ljung_box'][10]:,.0f}",
         fmt_p(autoc["returns_p_values"][10]), "Weak linear memory"],
        ["Ljung-Box, squared returns (lag 10)",
         f"{autoc['squared_returns_ljung_box'][10]:,.0f}",
         fmt_p(autoc["squared_returns_p_values"][10]), "ARCH effects present"],
        ["One-way ANOVA, sector means", f"{anova['f_stat']:.2f}",
         fmt_p(anova["p_value"]),
         "Equal sector means " + (
             "rejected" if anova["p_value"] < 0.05 else "not rejected")],
    ]
    TBL(
        ["Test", "Statistic", "p-value", "Reading"],
        test_rows,
        "Hypothesis-testing suite at the 5% significance level.",
        ratios=[0.34, 0.2, 0.2, 0.26], aligns=["L", "R", "C", "L"],
    )
    P(
        f"The suite yields an unambiguous verdict on each modelling decision. "
        f"{norm['n_reject']} of {norm['n_tickers']} tickers fail Jarque-Bera with pooled "
        f"excess kurtosis of {norm['pooled']['excess_kurtosis']:.1f}: daily returns are "
        f"fat-tailed, so non-parametric tree ensembles are preferred over Gaussian "
        f"assumptions. ADF separates prices from returns cleanly, licensing return-based "
        f"modelling. Ljung-Box finds little linear autocorrelation in returns - the "
        f"weak-form efficiency benchmark the classifier must beat - but overwhelming "
        f"autocorrelation in squared returns, i.e. volatility clustering, which is exactly "
        f"the structure the volatility features exploit. The sector ANOVA "
        f"(F = {anova['f_stat']:.2f}) determines whether sector dummies earn their place "
        f"in the feature set."
    )
    FIG("qq_plot",
        "QQ-plot of pooled daily returns against Gaussian quantiles: the S-shaped "
        "deviation in the tails mirrors the excess kurtosis measured above.")
    jb_view = (
        norm["per_ticker"]
        .sort_values("excess_kurtosis", ascending=False)
        .head(6)
    )
    jb_rows = [
        [esc(r["ticker"]), f"{r['jb_stat']:,.0f}", fmt_p(r["p_value"]),
         f"{r['skew']:.2f}", f"{r['excess_kurtosis']:.1f}"]
        for _, r in jb_view.iterrows()
    ]
    TBL(
        ["Ticker", "JB statistic", "p-value", "Skew", "Excess kurtosis"],
        jb_rows,
        "Six tickers with the heaviest tails (Jarque-Bera).",
        ratios=[0.2, 0.26, 0.22, 0.16, 0.16], aligns=["C", "R", "C", "R", "R"],
    )

    # =========================================================================
    # 6. Feature engineering and methodology
    # =========================================================================
    H1(6, "Feature Engineering and Methodology")
    P(
        "The label construction is the single most leakage-sensitive step in any "
        "panel study, so the pipeline pins it down explicitly: the target is the sign "
        f"of the forward {par['horizon']}-trading-day return, computed as "
        "<i>close(t + horizon) / close(t) - 1 > 0</i> via a per-ticker "
        "<i>shift(-horizon)</i>. Labels are sampled every "
        f"{par['sampling_step']} trading days so that consecutive windows do not "
        "overlap, which keeps cross-validation folds honest - overlapping labels would "
        "inflate apparent significance. All twenty technical features - and the "
        "ten sector one-hot columns, fixed by the static universe definition - "
        "use only information available at the close of the labelling date: "
        "every indicator is either "
        "backward-looking (lags, rolling windows) or contemporaneous (today's range, "
        "volume and close location). The invariance is pinned by a unit test that "
        "truncates the future and asserts no feature value changes."
    )
    ff_rows = [
        [esc(fam), esc(desc), str(n), esc(ex)]
        for fam, desc, n, ex in context["feature_families"]
    ]
    TBL(
        ["Feature family", "Rationale", "Count", "Representative features"],
        ff_rows,
        "Feature families in the modelling dataset (all causally computed).",
        ratios=[0.18, 0.34, 0.08, 0.4], aligns=["L", "L", "C", "L"],
    )
    STATS([
        (f"{ds['rows']:,}", "model rows"),
        (f"{ds['tickers']}", "tickers"),
        (f"{ds['n_features']}", "features"),
        (f"{ds['up_period_share']:.1%}", "up-period share"),
    ])
    P(
        "Methodological guardrails carry through model selection as well. The dataset "
        f"is split chronologically with the final {par['test_ratio']:.0%} of dates "
        "reserved as an untouched hold-out, and model comparison inside the training "
        f"period uses {par['cv_folds']}-fold TimeSeriesSplit, so no fold ever trains on "
        "the future to predict the past. Class balance (the up-period share above) is "
        "reported alongside every accuracy figure so that the baseline is always "
        "explicit, and predicted probabilities - not hard labels - flow into the "
        "backtest, letting the economic layer choose its own confidence threshold."
    )

    # =========================================================================
    # 7. Machine-learning models
    # =========================================================================
    H1(7, "Machine-Learning Models")
    P(
        "Three candidate classifiers span the interpretability-performance spectrum: "
        "L2-regularised logistic regression as the interpretable baseline, random "
        "forest as the bagged-tree workhorse, and XGBoost as the gradient-boosted "
        "tabular benchmark. All three are trained on identical features and evaluated "
        "twice - time-series cross-validation inside the training window for model "
        "selection, then once on the chronological hold-out for the headline numbers."
    )
    cv_rows = [
        [esc(r["model"]), f"{r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f}",
         f"{r['cv_accuracy_mean']:.4f}"]
        for _, r in cv_df.iterrows()
    ]
    TBL(
        ["Model", "CV ROC-AUC (mean ± std)", "CV accuracy"],
        cv_rows,
        f"{par['cv_folds']}-fold TimeSeriesSplit cross-validation (training period only).",
        ratios=[0.3, 0.42, 0.28], aligns=["L", "C", "C"],
    )
    tm_rows = []
    for name, m in mod["test_metrics"].items():
        tm_rows.append([
            esc(name) + (" <b>(best)</b>" if name == best_name else ""),
            f"{m['accuracy']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}",
            f"{m['f1']:.4f}", f"{m['roc_auc']:.4f}",
        ])
    TBL(
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        tm_rows,
        f"Hold-out test metrics (final {par['test_ratio']:.0%} of dates). "
        f"Up-period base rate: {best['majority_class_share']:.1%}.",
        ratios=[0.28, 0.144, 0.144, 0.144, 0.144, 0.144],
        aligns=["L", "R", "R", "R", "R", "R"],
    )
    FIG("roc_curves",
        "ROC curves of all three models on the hold-out test set; the best model is "
        "drawn with the heavier line.")
    FIG("confusion_matrix",
        f"Confusion matrix of the {best_name} classifier at the 0.5 decision threshold.")
    FIG("feature_importance",
        f"Top-15 feature importances for the best tree ensemble; volatility, trend and "
        f"volume families dominate.")
    P(
        f"The honest reading of these tables matters more than the ranking itself. "
        f"Cross-validated AUCs hover near {cv_df['cv_roc_auc_mean'].min():.3f}-"
        f"{cv_df['cv_roc_auc_mean'].max():.3f} - statistically distinguishable from a "
        f"coin flip only marginally during the turbulent training window - while the "
        f"chronologically later hold-out period is kinder: the best model reaches "
        f"{best['roc_auc']:.3f} ROC-AUC and {best['accuracy']:.1%} accuracy against a "
        f"{best['majority_class_share']:.1%} base rate. An edge this thin is exactly "
        f"what efficient-market theory predicts for daily equity data; the interesting "
        f"question is whether it survives costs, which Section 8 answers. The "
        f"importance ranking at least is stable: realised volatility, Bollinger "
        f"positioning and trend ratios consistently outrun raw momentum lags, mirroring "
        f"the ARCH evidence from the Ljung-Box tests."
    )

    # =========================================================================
    # 8. Backtest and economic significance
    # =========================================================================
    H1(8, "Backtest and Economic Significance")
    P(
        f"Statistical metrics are necessary but not sufficient: a signal only matters "
        f"if it survives costs. The backtest converts hold-out probabilities into a "
        f"trading rule with three parameters fixed in advance - go long in every ticker "
        f"whose predicted P(up) is at least {par['threshold']:.2f} at each "
        f"{par['horizon']}-day rebalance, hold an equal-weight book of the selected "
        f"names for the horizon, and charge {par['cost_bps']} basis points per unit of "
        f"turnover. The strategy is compared against {uni['benchmark']} buy-and-hold "
        f"measured over identical rebalance windows, so the comparison is strictly "
        f"like-for-like. All performance statistics are annualised with "
        f"{252 / par['horizon']:.1f} periods per year and a "
        f"{par.get('risk_free_rate', 2.0):.0f}% risk-free rate."
    )
    bt_rows = [
        ["<b>ML strategy (net)</b>",
         f"{strategy['total_return_pct']:+.1f}", f"{strategy['cagr_pct']:+.1f}",
         f"{strategy['ann_vol_pct']:.1f}", f"<b>{strategy['sharpe']:.2f}</b>",
         f"{strategy['max_drawdown_pct']:.1f}", f"{strategy['hit_rate_pct']:.1f}"],
        [f"{uni['benchmark']} buy {_AMP} hold",
         f"{benchmark['total_return_pct']:+.1f}", f"{benchmark['cagr_pct']:+.1f}",
         f"{benchmark['ann_vol_pct']:.1f}", f"{benchmark['sharpe']:.2f}",
         f"{benchmark['max_drawdown_pct']:.1f}", f"{benchmark['hit_rate_pct']:.1f}"],
    ]
    TBL(
        ["Strategy", "Total ret (%)", "CAGR (%)", "Ann. vol (%)", "Sharpe",
         "Max DD (%)", "Hit rate (%)"],
        bt_rows,
        f"Out-of-sample performance, {par['horizon']}-day rebalancing, "
        f"{par['cost_bps']} bps per turnover unit.",
        ratios=[0.26, 0.13, 0.12, 0.13, 0.12, 0.12, 0.12],
        aligns=["L", "R", "R", "R", "R", "R", "R"],
    )
    FIG("backtest",
        f"Out-of-sample equity curves: ML signal strategy net of costs versus "
        f"{uni['benchmark']} buy-and-hold over identical rebalance windows.")
    P(
        f"The mechanism behind the risk-adjusted edge is selectivity rather than "
        f"stock-picking brilliance. The strategy is invested on average "
        f"{strategy.get('avg_exposure_pct', float('nan')):.1f}% of the time - it "
        f"sits in cash whenever the classifier is not confident - so it skips a "
        f"meaningful share of the worst drawdown windows while still capturing most "
        f"of the up-periods. The result is a maximum drawdown of "
        f"{strategy['max_drawdown_pct']:.1f}% versus "
        f"{benchmark['max_drawdown_pct']:.1f}% for the benchmark at comparable "
        f"annualised volatility, which is what lifts the Sharpe ratio from "
        f"{benchmark['sharpe']:.2f} to {strategy['sharpe']:.2f}. Hit rate - the share "
        f"of profitable holding windows - ends at {strategy['hit_rate_pct']:.1f}%, "
        f"and average turnover per rebalance is "
        f"{strategy.get('avg_turnover_pct_per_period', float('nan')):.1f}%, keeping "
        f"total friction well inside the cost budget."
    )

    # =========================================================================
    # 9. Conclusions, limitations and reproducibility
    # =========================================================================
    H1(9, "Conclusions, Limitations and Reproducibility")
    P(
        f"This project set out to build, document and defend a complete quantitative "
        f"research workflow on real market data, and the exercise supports four "
        f"conclusions. First, the statistical foundations behave exactly as the "
        f"textbook predicts - fat tails, stationary returns, clustered volatility - "
        f"which validates both the cleaning pipeline and the feature design. Second, "
        f"predictive edge at the weekly horizon is real but thin: ROC-AUC of "
        f"{best['roc_auc']:.3f} on the hold-out is consistent with a market that is "
        f"nearly, but not perfectly, efficient. Third, the edge is economically "
        f"monetisable mainly through timing selectivity: staying in cash when "
        f"confidence is low converts a marginal statistical signal into a "
        f"{strategy['sharpe'] - benchmark['sharpe']:+.2f} Sharpe improvement and a "
        f"{strategy['max_drawdown_pct'] - benchmark['max_drawdown_pct']:+.1f} "
        f"percentage-point drawdown reduction versus buy-and-hold. Fourth, the "
        f"engineering discipline - caching, SQL layering, leakage tests, "
        f"deterministic seeds - is what makes such claims auditable at all."
    )
    P("The limitations are stated as plainly as the results:", "body")
    BUL([
        f"The universe is the current {SPX_P} membership, so survivorship bias flatters "
        "long-horizon return statistics; delisted names are absent.",
        "Predicted probabilities, thresholds and hyperparameters were fixed a priori, "
        "but the evaluation period is a single market regime - out-of-sample behaviour "
        "in a prolonged bear market remains untested.",
        "The cost model (2 bps per turnover unit, no market impact or slippage) suits "
        "liquid large caps only; the same rule would not scale to small caps.",
        "The strategy is long-only with cash as the alternative; a short leg, hedges "
        "or volatility targeting could change the risk profile materially.",
        "Close-to-close execution assumes fills at the same close used to compute "
        "features - realistic for a weekly cadence, but an approximation nonetheless.",
        "CV-period AUC near the coin-flip level is an honest warning: the hold-out "
        "uplift may partly reflect a friendlier regime rather than pure skill.",
    ])
    P(
        "Reproducibility is treated as a feature, not an afterthought. The repository "
        "carries the full pipeline behind this report - data acquisition, SQLite "
        "warehouse, cleaning, EDA, statistical tests, feature engineering, modelling "
        "and backtest - runnable end-to-end with <i>python run_pipeline.py</i> and "
        "pinned in <i>requirements.txt</i>. Random seeds are fixed, the Parquet cache "
        "makes re-runs byte-stable, all figures and tables in this document are "
        "regenerated from <i>results.json</i>, and the unit-test suite "
        "(<i>pytest tests/</i>) pins the no-look-ahead invariants that make the "
        "conclusions defensible."
    )

    # ---- Build ---------------------------------------------------------------
    doc = TocDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=MARGIN_T, bottomMargin=MARGIN_B,
        title=f"{REPORT_TITLE} - {REPORT_SUBTITLE}",
        author="EquityLens Project",
        creator="equitylens pipeline (ReportLab)",
        subject=(
            "End-to-end quantitative equity research: SQL analytics, hypothesis "
            "testing, leakage-safe feature engineering, ML direction prediction and "
            "net-of-cost backtesting on the S&P 500 universe."
        ),
    )
    doc.multiBuild(
        story,
        onFirstPage=_make_cover_painter(context, fonts),
        onLaterPages=_make_furniture_painter(fonts, str(context["generated_at"])),
    )

    size_mb = output_path.stat().st_size / 1e6
    logger.info(
        "PDF report written to %s (%.2f MB, %d figures, %d tables)",
        output_path, size_mb, fig_n, tbl_n,
    )
    return str(output_path)
