#!/usr/bin/env python3
"""Generate a polished single-image leaderboard card for social posts.

Reads assets/v0.3/summary.json and writes assets/v0.3/leaderboard_social.png —
a landscape, tier-coloured horizontal bar chart designed to be legible as a
LinkedIn / Twitter feed image (unlike the dense README leaderboard plot).

Usage:  python3 scripts/social_card.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SUMMARY = REPO / "assets" / "v0.3.1" / "summary.json"
OUT = REPO / "assets" / "v0.3.1" / "leaderboard_social.png"

# palette
INK = "#1a1a2e"
MUTE = "#6b7280"
TOP = "#2e9e6b"   # green  — top tier (<1%)
MID = "#e0a93b"   # amber  — mid tier
LOW = "#cb4f4f"   # red    — lagging  (7%+)
BG = "#ffffff"


def tier(rate: float) -> str:
    if rate < 1.0:
        return TOP
    if rate < 7.0:
        return MID
    return LOW


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    rows = []
    for model, v in summary["by_model"].items():
        ci = v["micro_broken_rate_completed_ci"]
        rows.append(
            {
                "model": model.split("/")[-1],
                "rate": v["micro_broken_rate_completed"] * 100,
                "lo": ci["lo"] * 100,
                "hi": ci["hi"] * 100,
            }
        )
    rows.sort(key=lambda r: r["rate"])
    n = len(rows)

    fig, ax = plt.subplots(figsize=(14.5, 8.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y = list(range(n))[::-1]  # rank 1 at the top
    rates = [r["rate"] for r in rows]
    colors = [tier(r) for r in rates]

    ax.barh(y, rates, height=0.64, color=colors, edgecolor="none", zorder=3)

    # Wilson CI whiskers — subtle
    for yi, r in zip(y, rows):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=INK, alpha=0.45,
                lw=1.5, zorder=4, solid_capstyle="round")
        for x in (r["lo"], r["hi"]):
            ax.plot([x, x], [yi - 0.12, yi + 0.12], color=INK, alpha=0.45, lw=1.5, zorder=4)

    # value labels past the CI whisker
    for yi, r in zip(y, rows):
        ax.text(r["hi"] + 0.55, yi, f'{r["rate"]:.1f}%', va="center", ha="left",
                fontsize=14.5, fontweight="bold", color=INK, zorder=5)

    # model labels (rank + name) on the left
    ax.set_yticks(y)
    ax.set_yticklabels([f'{i+1}.  {r["model"]}' for i, r in enumerate(rows)],
                       fontsize=14, color=INK)
    ax.tick_params(axis="y", length=0)

    # top-3 tie bracket + label, in the empty space right of the top-3 value labels
    x_br = 4.6
    yt, yb = y[0] + 0.4, y[2] - 0.4
    ax.plot([x_br, x_br], [yb, yt], color=MUTE, lw=1.7, zorder=5)
    for yy in (yt, yb):
        ax.plot([x_br, x_br - 0.18], [yy, yy], color=MUTE, lw=1.7, zorder=5)
    ax.text(x_br + 0.4, y[1], "Top 3 — a statistical tie\n(95% CIs overlap)",
            fontsize=13, color=MUTE, va="center", style="italic")

    # axes cosmetics
    ax.set_xlim(0, 18)
    ax.set_ylim(-0.7, n - 0.3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTE)
    ax.tick_params(axis="x", colors=MUTE, labelsize=12.5)
    ax.set_xticks(range(0, 19, 2))
    ax.set_xticklabels([f"{x}%" for x in range(0, 19, 2)])
    ax.grid(axis="x", color="#e5e7eb", lw=0.9, zorder=0)
    ax.set_axisbelow(True)

    # titles — anchored 10% in from the left edge, sitting in the expanded top margin
    fig.text(0.063, 0.917, "Do frontier LLMs honor the rules under pressure?",
             fontsize=25, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.063, 0.860,
             "RefuseBench v0.3.1 — rule-violation rate when a buried policy constraint gets "
             "inconvenient.  Lower is better.",
             fontsize=14.5, color=MUTE, ha="left", va="top")
    fig.text(0.063, 0.816,
             "11 models  ·  10 realistic policy scenarios  ·  330 responses  ·  129 rules",
             fontsize=12.5, color=MUTE, ha="left", va="top")

    # tier legend — top-right empty zone
    handles = [
        plt.Line2D([0], [0], marker="s", ls="", ms=14, mfc=TOP, mec="none",
                   label="Top tier  (<1%)"),
        plt.Line2D([0], [0], marker="s", ls="", ms=14, mfc=MID, mec="none",
                   label="Mid  (1–6%)"),
        plt.Line2D([0], [0], marker="s", ls="", ms=14, mfc=LOW, mec="none",
                   label="Lagging  (7%+)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=13,
              handletextpad=0.3, borderpad=0.6, labelcolor=INK,
              bbox_to_anchor=(0.999, 0.99))

    # footer — two methodology lines (left) + repo URL (right), inside the expanded bottom margin
    fig.text(0.063, 0.106,
             "Judged by a 3-vendor LLM committee · calibrated against 150 blind human "
             "labels (Cohen's κ 0.74–0.79).",
             fontsize=10.5, color=MUTE, ha="left", va="bottom")
    fig.text(0.063, 0.070,
             "All models engaged 90–100% of the time — refusals are not inflating these scores.",
             fontsize=10.5, color=MUTE, ha="left", va="bottom")
    fig.text(0.937, 0.084, "github.com/gimocimo/RefuseBench",
             fontsize=12.5, color=INK, ha="right", va="bottom", fontweight="bold")

    fig.subplots_adjust(left=0.255, right=0.925, top=0.75, bottom=0.165)
    fig.savefig(OUT, facecolor=BG)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
