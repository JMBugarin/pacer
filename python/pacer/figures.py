"""README figures from the test-window summary and the hazard-model reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .report import CAPS, TRACE_LABEL, TRACES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, GRAY_DARK = "#8f8e88", "#3d3c39"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def hit_vs_capacity(summary: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), facecolor=SURFACE)
    xs = [100 * c for c in CAPS]
    series = [
        ("Belady (offline bound)", "belady", GRAY_DARK, "--", None),
        ("Sampled Belady (oracle)", "sampled_belady", GRAY, ":", None),
        ("Pacer", "pacer_fixed", BLUE, "-", "o"),
        ("Best tuned heuristic", "best_heuristic_hit", ORANGE, "-", "s"),
        ("ATC'25 workload-aware", "wa", AQUA, "-", "^"),
        ("LRU (vLLM default)", "lru", GRAY, "-", "D"),
    ]
    for ax, t in zip(axes, TRACES, strict=True):
        style(ax)
        cells = summary.get(t, {})
        for name, key, color, ls, marker in series:
            ys = []
            for c in CAPS:
                cell = cells.get(str(c), {})
                v = cell["all"].get("wa") if key == "wa" else cell.get(key)
                ys.append(100 * v if v is not None else float("nan"))
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=ls,
                linewidth=2,
                marker=marker,
                markersize=6 if marker else 0,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=name,
            )
        ax.set_xscale("log")
        ax.set_xticks(xs, [f"{x:g}%" for x in xs])
        ax.minorticks_off()
        ax.set_title(TRACE_LABEL[t], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("cache size (% of unique blocks)", color=INK_2, fontsize=9)
    axes[0].set_ylabel("block hit ratio, test hour (%)", color=INK_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def oracle_ladder(summary: dict, trace: str, out: Path) -> None:
    rungs = [
        ("LRU", "lru", GRAY, None),
        ("ATC'25", "wa", AQUA, None),
        ("Pacer", "pacer-fixed", BLUE, None),
        ("oracle: reused within 60 s", "oracle-within-60", GRAY, "//"),
        ("oracle: reused within 300 s", "oracle-within-300", GRAY, ".."),
        ("oracle: never reused", "oracle-dead", GRAY, "xx"),
        ("sampled Belady", "sampled-belady", "#6b6a66", None),
        ("Belady", "belady", GRAY_DARK, None),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax)
    width = 0.8 / len(rungs)
    for i, (name, key, color, hatch) in enumerate(rungs):
        xs, ys = [], []
        for j, c in enumerate(CAPS):
            cell = summary[trace][str(c)]
            xs.append(j + (i - len(rungs) / 2 + 0.5) * width)
            ys.append(100 * cell["all"][key])
        ax.bar(
            xs,
            ys,
            width=width - 0.01,
            color=color,
            edgecolor=SURFACE,
            linewidth=1,
            hatch=hatch,
            label=name,
        )
    ax.set_xticks(range(len(CAPS)), [f"{100 * c:g}% cache" for c in CAPS])
    ax.set_ylabel("block hit ratio, test hour (%)", color=INK_2, fontsize=9)
    ax.set_title(f"{TRACE_LABEL[trace]}: where the headroom is", color=INK, fontsize=11, loc="left")
    ax.legend(
        frameon=False,
        fontsize=8,
        labelcolor=INK_2,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def auc_chart(reports: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 2.8), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ys = list(range(len(TRACES)))[::-1]
    for y, t in zip(ys, TRACES, strict=True):
        w = reports[t]["workload_exp"]["auc"]
        g = reports[t]["gbdt"]["auc"]
        ax.plot([w, g], [y, y], color=GRID, linewidth=2, zorder=1)
        ax.scatter([w], [y], s=70, color=AQUA, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        ax.scatter([g], [y], s=70, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        ax.text(w - 0.008, y + 0.22, f"{w:.3f}", ha="center", fontsize=8, color=INK_2)
        ax.text(g + 0.008, y + 0.22, f"{g:.3f}", ha="center", fontsize=8, color=INK_2)
    ax.set_yticks(ys, [TRACE_LABEL[t] for t in TRACES])
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.6, len(TRACES) - 0.3)
    ax.set_xlabel("held-out AUC (0.5 = chance)", color=INK_2, fontsize=9)
    ax.set_title("Predicting reuse within (a, a + h]", color=INK, fontsize=11, loc="left")
    ax.scatter([], [], s=50, color=AQUA, label="ATC'25 workload model")
    ax.scatter([], [], s=50, color=BLUE, label="Pacer hazard model")
    ax.legend(
        frameon=False,
        fontsize=8,
        labelcolor=INK_2,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.3),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/final_summary.json")
    ap.add_argument("--hazard", default="results/v2/fast_{trace}.json")
    ap.add_argument("--out", default="docs/images")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(a.summary).read_text())
    hit_vs_capacity(summary, out / "hit-vs-capacity.png")
    oracle_ladder(summary, "traceA", out / "oracle-ladder-traceA.png")
    reports = {t: json.loads(Path(a.hazard.format(trace=t)).read_text()) for t in TRACES}
    auc_chart(reports, out / "hazard-auc.png")


if __name__ == "__main__":
    main()
