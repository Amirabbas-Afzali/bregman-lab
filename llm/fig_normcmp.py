"""fig_normcmp.py — Canonical-vs-Amari Arena-Hard result per divergence (fL3).

Two stacked panels sharing the divergence axis:
  top    — Arena-Hard v0.1 win rate against the fixed baseline gpt-4-0314: Canonical (dark red,
           f'(1)=f''(1)) vs Amari (grey, f'(1)=0), with 95% bootstrap CIs.
  bottom — the direct Canonical-vs-Amari win rate as a box per divergence, read straight off the
           prompt-level bootstrap in results/bench/arena_v01/h2h_normcmp/*_raw.json: median line at
           the measured win rate, box at the bootstrap 25th-75th percentile, whiskers at its
           2.5th-97.5th (the 95% CI). 50% would be a tie; every divergence sits well above it, so
           the axis is zoomed past it.
Judge: gpt-4.1-mini-2025-04-14 (validated ≈ gpt-4.1), 500 Arena-Hard v0.1 prompts.

Divergences are in the canonical REGKEYS order (RKL, α-div, FKL, JS, Hel, χ²), matching the tabular figures.

Run from llm/:  python fig_normcmp.py   ->  results/stageB_normcmp_wr_h2h.{png,pdf}
Then `python export_for_paper.py` from the repo root copies it to figure4overleaf/fL3_normcmp_wr_h2h.*
together with its provenance sidecar.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from divergences import COLORS   # per-divergence palette shared across all figures

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "mathtext.fontset": "cm", "axes.linewidth": 0.9})

DIVS = ["RKL", "α-div", "FKL", "JS", "Hel", "χ²"]       # canonical REGKEYS order (kl,adiv,rkl,js,hel,chi2)
BAR_KEYS = ["kl", "adiv", "rkl", "js", "hel", "chi2"]   # DIVS -> divergences.COLORS key (kl=RKL, rkl=FKL)

# A: (amari_wr, (lo,hi) offsets) · B: (canon_wr, (lo,hi) offsets) — Arena-Hard v0.1 vs gpt-4-0314,
# from arena-hard-auto show_result.py (Bradley-Terry bootstrap CI).
DATA = {
    "RKL":   ((8.4,  (1.1, 1.0)), (14.4, (1.4, 1.3))),
    "α-div": ((9.7,  (1.1, 1.0)), (13.0, (1.3, 1.4))),
    "FKL":   ((10.1, (1.2, 1.1)), (13.5, (1.3, 1.1))),
    "JS":    ((10.3, (1.1, 1.3)), (13.7, (1.3, 1.4))),
    "Hel":   ((9.8,  (1.2, 1.0)), (12.5, (1.2, 1.1))),
    "χ²":    ((7.2,  (1.0, 0.9)), (13.5, (1.3, 1.2))),
}

# C: the direct head-to-head, taken from pairwise_h2h.py's own bootstrap replicates rather than a
# summary, so the box shows the real distribution. Files hold AMARI's win rate; canonical is 1 - that.
H2H_DIR = "results/bench/arena_v01/h2h_normcmp"
H2H_KEY = {"RKL": "kl", "α-div": "adiv", "FKL": "fkl", "JS": "js", "Hel": "hel", "χ²": "chi2"}


def pct(xs, p):
    """p-quantile of a sorted list, linearly interpolated."""
    i = p * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - (i - lo)) + xs[hi] * (i - lo)


def h2h_box(div):
    k = H2H_KEY[div]
    summ = json.load(open(f"{H2H_DIR}/h2h_{k}_amari_vs_canon_mini2.json"))
    boot = json.load(open(f"{H2H_DIR}/h2h_{k}_amari_vs_canon_mini2_raw.json"))["boot"]
    canon = sorted(100.0 * (1.0 - b) for b in boot)      # amari replicate -> canonical replicate
    return {"med": round(100 - summ["win_rate"], 1), "q1": pct(canon, 0.25), "q3": pct(canon, 0.75),
            "whislo": pct(canon, 0.025), "whishi": pct(canon, 0.975), "label": div}


C_AMARI, C_CANON = "#6b7280", "#b0224b"

# equal-height panels, near-flush so the shared divergence axis reads as one figure
fig = plt.figure(figsize=(10.6, 5.0))
gs = fig.add_gridspec(2, 1, height_ratios=(1, 1), hspace=0.06,
                      left=0.085, right=0.985, top=0.825, bottom=0.085)
axT = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1], sharex=axT)
x = list(range(len(DIVS)))

# top: win rate against the fixed gpt-4-0314 baseline (no connector — the pair reads from the colour)
for i, d in enumerate(DIVS):
    a, c = DATA[d]
    aw, (alo, ahi) = a
    cw, (clo, chi) = c
    axT.errorbar(i, aw, yerr=[[alo], [ahi]], fmt="o", ms=8, color=C_AMARI, ecolor=C_AMARI,
                 elinewidth=1.3, capsize=3, zorder=5)
    axT.errorbar(i, cw, yerr=[[clo], [chi]], fmt="o", ms=8, color=C_CANON, ecolor=C_CANON,
                 elinewidth=1.3, capsize=3, zorder=6)
    axT.text(i - 0.10, aw, f"{aw:.1f}", color=C_AMARI, fontsize=9, va="center", ha="right")
    axT.text(i - 0.10, cw, f"{cw:.1f}", color=C_CANON, fontsize=9, va="center", ha="right", fontweight="bold")

# bottom: Canonical vs Amari head-on, one box per divergence from the bootstrap replicates
stats = [h2h_box(d) for d in DIVS]
cols = [COLORS[k] for k in BAR_KEYS]

bp = axB.bxp(stats, positions=x, widths=0.52, patch_artist=True, showfliers=False, zorder=2)
for k, (box, med, col) in enumerate(zip(bp["boxes"], bp["medians"], cols)):
    box.set(facecolor=col, alpha=0.24, edgecolor=col, linewidth=1.4)
    med.set(color=col, linewidth=2.4, alpha=1.0)
    for art in (bp["whiskers"][2 * k], bp["whiskers"][2 * k + 1], bp["caps"][2 * k], bp["caps"][2 * k + 1]):
        art.set(color=col, linewidth=1.4, alpha=0.9)
for st, col in zip(stats, cols):
    axB.text(DIVS.index(st["label"]), st["whishi"] + 0.5, f"{st['med']:.1f}", color=col, fontsize=9,
             va="bottom", ha="center", fontweight="bold")

# judge provenance in each panel's empty top-right strip (text only, no frame)
axT.text(0.99, 0.97, "Judge: gpt-4.1-mini-2025-04-14",
         transform=axT.transAxes, ha="right", va="top", fontsize=8.5, color="#666666")
axB.text(0.99, 0.97, "Judge: gpt-4.1-mini-2025-04-14",
         transform=axB.transAxes, ha="right", va="top", fontsize=8.5, color="#666666")

# integer ticks every 2 pts (7-15) with a little headroom so no CI cap is clipped
axT.set_ylim(6.0, 16.4); axT.set_yticks([7, 9, 11, 13, 15])
axB.set_ylim(52, 70);    axB.set_yticks([55, 60, 65, 70])   # zoomed to the boxes (50% tie is off-scale)
axT.set_xlim(-0.6, len(DIVS) - 0.4)
axT.set_ylabel("Qwen 1.7B vs gpt-4\nWin Rate (%)", fontsize=11)
axB.set_ylabel("Qwen 1.7B\nCanonical vs Amari\nWin Rate (%)", fontsize=11)
axB.set_xticks(x); axB.set_xticklabels(DIVS, fontsize=12)
plt.setp(axT.get_xticklabels(), visible=False)
axT.tick_params(axis="x", length=0)
for ax in (axT, axB):
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)

legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CANON, ms=10,
                 label="Canonical (Ours)  $f'(1)=f''(1)$"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AMARI, ms=10,
                 label="Amari  $f'(1)=0$")]
# title on top, legend on its own band beneath it — both clear of the axes frame
fig.suptitle("Arena-Hard Results", fontsize=13.5, y=0.985)
fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, fontsize=11,
           frameon=True, framealpha=0.95, edgecolor="#bbbbbb", columnspacing=2.2, handletextpad=0.6,
           borderpad=0.6, labelspacing=0.5)

out = "results/stageB_normcmp_wr_h2h"
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
print(f"[saved] {out}.png/.pdf")
