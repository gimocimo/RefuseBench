#!/usr/bin/env python3
"""Generate the paper-specific figures (real data, for the skeleton PDF).

Two figures the README doesn't already have a standalone PNG for:
  * multi_turn_degradation.png — single-turn vs multi-turn violation rate per
    scenario + pooled, with bootstrap CI whiskers on the multi-turn bars.
  * embedding_penalty.png — per-model embedding penalty (embedded - foregrounded)
    with 95% bootstrap CIs; bars whose CI excludes zero are highlighted.

Reads the committed v0.6 / v0.3.1 analysis JSON. Writes into paper/figs/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

INK = "#1a1a2e"
SINGLE = "#9bb7d4"
MULTI = "#cb4f4f"
SIG = "#2e9e6b"
MUTE = "#9aa0a6"


def multi_turn_fig():
    d = json.loads((REPO / "assets" / "v0.6" / "multi_turn_study.json").read_text())
    pd = d["pressure_degradation"]
    order = ["dba_latency_gate", "code_review_under_deadline", "customer_support_escalation"]
    labels = ["dba_latency_gate", "code_review", "customer_support", "POOLED"]
    single = [pd[s]["single_turn_rate"] * 100 for s in order] + [d["overall"]["single_turn_rate"] * 100]
    multi = [pd[s]["multi_turn_rate"] * 100 for s in order] + [d["overall"]["multi_turn_rate"] * 100]
    ci = [pd[s]["delta_ci_pp"] for s in order] + [d["overall"]["delta_ci_pp"]]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(labels))
    w = 0.38
    ax.bar([i - w / 2 for i in x], single, w, label="single-turn", color=SINGLE)
    mbars = ax.bar([i + w / 2 for i in x], multi, w, label="multi-turn (final-state)", color=MULTI)
    for i, (m, dlt) in enumerate(zip(multi, [pd[s]["delta_pp"] for s in order] + [d["overall"]["delta_pp"]])):
        ax.text(i + w / 2, m + 0.6, f"+{dlt:.0f}pp", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Violation rate among completed (%)")
    ax.set_title("Multi-turn pressure degrades policy compliance (shared base rules)", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "multi_turn_degradation.png", dpi=150)
    print("wrote", FIGS / "multi_turn_degradation.png")


def embedding_penalty_fig():
    d = json.loads((REPO / "assets" / "v0.3.1" / "baseline_study_contemporaneous.json").read_text())
    rows = sorted(d["per_model"], key=lambda r: r["penalty_pp"], reverse=True)
    names = [r["model"].split("/")[-1] for r in rows]
    pen = [r["penalty_pp"] for r in rows]
    los = [r["penalty_ci_pp"][0] for r in rows]
    his = [r["penalty_ci_pp"][1] for r in rows]
    colors = [SIG if (lo > 0 or hi < 0) else MUTE for lo, hi in zip(los, his)]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = range(len(names))
    err_lo = [p - lo for p, lo in zip(pen, los)]
    err_hi = [hi - p for p, hi in zip(pen, his)]
    ax.barh(list(y), pen, color=colors, xerr=[err_lo, err_hi],
            error_kw=dict(ecolor="#555", lw=1, capsize=3))
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Embedding penalty = violation rate(embedded) − violation rate(foregrounded), pp")
    ax.set_title("Per-model embedding penalty (green = 95% CI excludes zero)", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "embedding_penalty.png", dpi=150)
    print("wrote", FIGS / "embedding_penalty.png")


if __name__ == "__main__":
    multi_turn_fig()
    embedding_penalty_fig()
