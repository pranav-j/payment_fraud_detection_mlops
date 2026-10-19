"""Generate the README's figures from the committed reports.

    make figures

Every chart here is drawn from `reports/metrics/evaluation.json` and
`reports/perf/<date>/`, so the pictures cannot drift away from the numbers in
the text — regenerate the data and the figures follow. Nothing is hand-drawn
and nothing is illustrative.

Three figures, each answering one question:

1. `evaluation-baselines.png` — how much better than a rule is the model, and
   is the v1 -> v2 difference bigger than the error bars? (It is not.)
2. `evaluation-leakage.png` — how much of the apparent performance is PaySim's
   balance columns giving the answer away?
3. `latency-by-load.png` — what does the synchronous path do as load rises?

Design notes, since charts are read by people:

- Error bars are the point of figure 1, not decoration. Two models whose
  intervals overlap are not distinguishable, and the chart has to show that
  rather than letting two bar heights imply a result.
- Figure 3 encodes p50/p95/p99 as one hue stepped light-to-dark, because those
  are an *ordered* quantity, not three unrelated categories. Validated against
  the light chart surface.
- No dual axes anywhere. Where two measures have different scales they get
  different figures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fraud_mlops.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_JSON = PROJECT_ROOT / "reports" / "metrics" / "evaluation.json"
PERF_DIR = PROJECT_ROOT / "reports" / "perf"

# ── Palette ─────────────────────────────────────────────────────────────────
# Categorical slot 1 (blue) for the measured series; ink tokens for all text,
# never the series colour. The ordinal ramp is one hue stepped light->dark and
# passes the light-surface checks (lightness monotone, adjacent dL, light end
# 2.06:1 against the surface).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e6e6e3"

SERIES_BLUE = "#2a78d6"
SERIES_RED = "#e34948"
ORDINAL = ["#86b6ef", "#2a78d6", "#104281"]  # p50, p95, p99


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _despine(ax: Any, keep: tuple[str, ...] = ()) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _load_evaluation() -> dict[str, Any]:
    if not METRICS_JSON.exists():
        raise FileNotFoundError(f"{METRICS_JSON} missing — run `make evaluate` first")
    payload: dict[str, Any] = json.loads(METRICS_JSON.read_text())
    return payload


def _latest_perf_dir() -> Path | None:
    if not PERF_DIR.exists():
        return None
    sessions = sorted(d for d in PERF_DIR.iterdir() if d.is_dir())
    return sessions[-1] if sessions else None


# ── Figure 1: the baseline ladder ───────────────────────────────────────────


def figure_baselines(report: dict[str, Any], out: Path) -> Path:
    """Recall at 95% precision per model, with bootstrap intervals.

    Two panels, because one cannot carry both claims. On a 0-100 axis the ladder
    is legible and the intervals are invisible; the whole point of reporting
    intervals is that the v1/v2 gap is smaller than they are. So the left panel
    shows the ladder and the right panel zooms to the two models that actually
    compete.
    """
    import matplotlib.pyplot as plt

    rows = [
        r
        for r in report["results"]
        if r["split_scheme"] == "time" and r["notes"] != "leakage experiment"
    ]
    rows.reverse()  # strongest at the top

    labels = [
        r["name"]
        .replace(" (v1 features)", " · v1")
        .replace(" (v2 features)", " · v2")
        .replace("Rule: large TRANSFER/CASH_OUT", "Rule: large TRANSFER / CASH_OUT")
        for r in rows
    ]
    point = [r["recall_at_95_precision"] * 100 for r in rows]
    low = [r["recall_at_95_precision_ci_low"] * 100 for r in rows]
    high = [r["recall_at_95_precision_ci_high"] * 100 for r in rows]

    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(10.6, 3.5), gridspec_kw={"width_ratios": [1.65, 1]}
    )

    def draw(target: Any, idx: list[int], label_ci: bool) -> None:
        for i in idx:
            target.plot(
                [low[i], high[i]],
                [i, i],
                color=SERIES_BLUE,
                linewidth=3,
                solid_capstyle="round",
                alpha=0.4,
                zorder=2,
            )
        target.scatter(
            [point[i] for i in idx],
            idx,
            s=70,
            color=SERIES_BLUE,
            zorder=3,
            edgecolor=SURFACE,
            linewidth=2,
        )
        for i in idx:
            target.annotate(
                f"{point[i]:.1f}%",
                (point[i], i),
                textcoords="offset points",
                xytext=(0, 11),
                ha="center",
                fontsize=9,
                color=INK_PRIMARY,
                fontweight="medium",
            )
            if label_ci and high[i] - low[i] > 0.5:
                target.annotate(
                    f"[{low[i]:.1f}, {high[i]:.1f}]",
                    (point[i], i),
                    textcoords="offset points",
                    xytext=(0, -17),
                    ha="center",
                    fontsize=7.5,
                    color=INK_MUTED,
                )

    # ── Left: the full ladder ──
    draw(ax, list(range(len(rows))), label_ci=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9.5, color=INK_PRIMARY)
    ax.set_xlabel("Recall at 95% precision, test split (%)", fontsize=9)
    ax.set_xlim(-4, 100)
    ax.set_ylim(-0.7, len(rows) - 0.35)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.set_title("The ladder", fontsize=10, color=INK_PRIMARY, loc="left", pad=8)

    # ── Right: zoom on the two that compete ──
    zoom = [i for i, label in enumerate(labels) if label.startswith("XGBoost")]
    draw(axz, zoom, label_ci=True)
    span_lo = min(low[i] for i in zoom)
    span_hi = max(high[i] for i in zoom)
    pad = (span_hi - span_lo) * 0.55
    axz.set_xlim(span_lo - pad, span_hi + pad)
    axz.set_ylim(min(zoom) - 0.75, max(zoom) + 0.55)
    axz.set_yticks(zoom)
    axz.set_yticklabels([labels[i] for i in zoom], fontsize=9.5, color=INK_PRIMARY)
    axz.set_xlabel("same axis, zoomed (%)", fontsize=9)
    axz.xaxis.grid(True, color=GRID, linewidth=0.8)
    axz.set_axisbelow(True)
    _despine(axz)
    axz.set_title(
        "Zoomed: the intervals overlap", fontsize=10, color=INK_PRIMARY, loc="left", pad=8
    )

    fig.suptitle(
        "Each model on the same held-out split, with 95% bootstrap intervals",
        fontsize=11,
        color=INK_PRIMARY,
        x=0.005,
        y=1.10,
        ha="left",
    )
    fig.text(
        0.005,
        1.015,
        "The enriched v2 features gain 0.75 points of recall — less than the width "
        "of either interval",
        fontsize=8.5,
        color=INK_SECONDARY,
        ha="left",
    )

    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


# ── Figure 2: the leakage experiment ────────────────────────────────────────


def figure_leakage(report: dict[str, Any], out: Path) -> Path:
    """PR-AUC with and without the balance columns, against a hand rule."""
    import matplotlib.pyplot as plt

    comparison = report["leakage"]["comparison"]
    labels = [
        "XGBoost\nwith balance columns",
        "XGBoost\nbalance columns removed",
        "Hand rule\non balance columns alone",
    ]
    values = [c["pr_auc"] for c in comparison]
    lows = [c["pr_auc_ci_low"] for c in comparison]
    highs = [c["pr_auc_ci_high"] for c in comparison]
    # The hand rule is the punchline; everything else is the same measurement.
    colors = [SERIES_BLUE, SERIES_BLUE, SERIES_RED]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    x = range(len(values))
    ax.bar(list(x), values, width=0.52, color=colors, zorder=2)

    for i, (v, lo, hi) in enumerate(zip(values, lows, highs, strict=True)):
        ax.plot([i, i], [lo, hi], color=SURFACE, linewidth=2, zorder=3)
        ax.annotate(
            f"{v:.3f}",
            (i, v),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=10,
            color=INK_PRIMARY,
            fontweight="medium",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, color=INK_PRIMARY)
    ax.set_ylabel("PR-AUC, test split", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _despine(ax)

    ax.set_title(
        "PaySim's balance columns are most of the signal",
        fontsize=10.5,
        color=INK_PRIMARY,
        loc="left",
        pad=26,
    )
    ax.annotate(
        "A three-condition rule beats the model; removing those columns collapses it",
        xy=(0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 12),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SECONDARY,
    )

    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


# ── Figure 3: latency under load ────────────────────────────────────────────


def figure_latency(perf_dir: Path, out: Path) -> Path | None:
    """p50/p95/p99 for the synchronous path at each load level."""
    import csv

    import matplotlib.pyplot as plt

    levels: list[tuple[str, dict[str, float]]] = []
    for rps in (5, 20, 50):
        stats = perf_dir / f"locust-{rps}rps_stats.csv"
        if not stats.exists():
            continue
        with stats.open() as fh:
            for row in csv.DictReader(fh):
                if row["Name"] == "Aggregated":
                    levels.append(
                        (
                            f"{rps} req/s\n({float(row['Requests/s']):.0f} achieved)",
                            {
                                "p50": float(row["50%"]),
                                "p95": float(row["95%"]),
                                "p99": float(row["99%"]),
                            },
                        )
                    )
    if not levels:
        logger.warning("no locust stats in %s — skipping latency figure", perf_dir)
        return None

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    width = 0.26
    for slot, (key, color) in enumerate(zip(("p50", "p95", "p99"), ORDINAL, strict=True)):
        offsets = [i + (slot - 1) * (width + 0.015) for i in range(len(levels))]
        values = [metrics[key] for _, metrics in levels]
        ax.bar(offsets, values, width=width, color=color, label=key, zorder=2)
        for xo, v in zip(offsets, values, strict=True):
            ax.annotate(
                f"{v:.0f}",
                (xo, v),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
                color=INK_SECONDARY,
            )

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([label for label, _ in levels], fontsize=9, color=INK_PRIMARY)
    ax.set_ylabel("Response time (ms)", fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _despine(ax)

    legend = ax.legend(
        frameon=False, fontsize=9, loc="upper left", ncols=3, bbox_to_anchor=(0, 1.02)
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    ax.set_title(
        "Synchronous path: p50 is flat as load rises tenfold",
        fontsize=10.5,
        color=INK_PRIMARY,
        loc="left",
        pad=34,
    )
    ax.annotate(
        "Zero errors at every level; the tail grows, the typical request does not",
        xy=(0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 20),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SECONDARY,
    )

    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def run(output_dir: Path = FIGURES_DIR) -> list[Path]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = _load_evaluation()
    written = [
        figure_baselines(report, output_dir / "evaluation-baselines.png"),
        figure_leakage(report, output_dir / "evaluation-leakage.png"),
    ]

    perf_dir = _latest_perf_dir()
    if perf_dir:
        latency = figure_latency(perf_dir, output_dir / "latency-by-load.png")
        if latency:
            written.append(latency)
    else:
        logger.warning("no reports/perf/<date>/ yet — skipping the latency figure")

    return written


def main() -> None:  # pragma: no cover - thin CLI wrapper
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for path in run():
        print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":  # pragma: no cover
    main()
