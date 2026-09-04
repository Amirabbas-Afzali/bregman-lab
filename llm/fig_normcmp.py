"""fig_normcmp.py — Canonical-vs-Amari Arena-Hard result per divergence (fL3).

Two stacked panels sharing the divergence axis:
  top    — Arena-Hard v0.1 win rate against the fixed baseline gpt-4-0314: Canonical (dark red,
           f'(1)=f''(1)) vs Amari (grey, f'(1)=0), with 95% bootstrap CIs.
  bottom — the direct Canonical-vs-Amari win rate (50% = tie), bars coloured with each divergence's
           shared palette colour, 95% prompt-level bootstrap CI.
Judge: gpt-4.1-mini-2025-04-14 (validated ≈ gpt-4.1), 500 Arena-Hard v0.1 prompts.

Divergences are in the canonical REGKEYS order (RKL, α-div, FKL, JS, Hel, χ²), matching the tabular figures.

Run from llm/:  python fig_normcmp.py   ->  results/stageB_normcmp_wr_h2h.{png,pdf}
Then `python export_for_paper.py` from the repo root copies it to figure4overleaf/fL3_normcmp_wr_h2h.*
together with its provenance sidecar.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from divergences import COLORS   # per-divergence palette shared across all figures

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "mathtext.fontset": "cm", "axes.linewidth": 0.9})

DIVS = ["RKL", "α-div", "FKL", "JS", "Hel", "χ²"]       # canonical REGKEYS order (kl,adiv,rkl,js,hel,chi2)
BAR_KEYS = ["kl", "adiv", "rkl", "js", "hel", "chi2"]   # DIVS -> divergences.COLORS key (kl=RKL, rkl=FKL)

# A: (amari_wr, (lo,hi) offsets) · B: (canon_wr, (lo,hi) offsets) · C: (canon-vs-amari win%, (lo,hi) absolute CI)
DATA = {
    "RKL":   ((8.4,  (1.1, 1.0)), (14.4, (1.4, 1.3)), (63.7, (60.4, 67.1))),
    "α-div": ((9.7,  (1.1, 1.0)), (13.0, (1.3, 1.4)), (59.8, (56.3, 63.2))),
    "FKL":   ((10.1, (1.2, 1.1)), (13.5, (1.3, 1.1)), (57.6, (53.9, 60.8))),
    "JS":    ((10.3, (1.1, 1.3)), (13.7, (1.3, 1.4)), (58.2, (54.7, 61.6))),
    "Hel":   ((9.8,  (1.2, 1.0)), (12.5, (1.2, 1.1)), (58.1, (54.5, 61.8))),
    "χ²":    ((7.2,  (1.0, 0.9)), (13.5, (1.3, 1.2)), (61.8, (58.3, 65.4))),
}

C_AMARI, C_CANON = "#6b7280", "#b0224b"

# equal-height panels, near-flush so the shared divergence axis reads as one figure
fig = plt.figure(figsize=(10.6, 5.0))
gs = fig.add_gridspec(2, 1, height_ratios=(1, 1), hspace=0.06)
axT = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1], sharex=axT)
x = list(range(len(DIVS)))

# top: win rate against the fixed gpt-4-0314 baseline (no connector — the pair reads from the colour)
for i, d in enumerate(DIVS):
    a, c, _ = DATA[d]
    aw, (alo, ahi) = a
    cw, (clo, chi) = c
    axT.errorbar(i, aw, yerr=[[alo], [ahi]], fmt="o", ms=8, color=C_AMARI, ecolor=C_AMARI,
                 elinewidth=1.3, capsize=3, zorder=5)
    axT.errorbar(i, cw, yerr=[[clo], [chi]], fmt="o", ms=8, color=C_CANON, ecolor=C_CANON,
                 elinewidth=1.3, capsize=3, zorder=6)
    axT.text(i - 0.10, aw, f"{aw:.1f}", color=C_AMARI, fontsize=9, va="center", ha="right")
    axT.text(i - 0.10, cw, f"{cw:.1f}", color=C_CANON, fontsize=9, va="center", ha="right", fontweight="bold")

# bottom: Canonical vs Amari head-on, coloured per divergence; 50% = tie is the axis floor
for i, d in enumerate(DIVS):
    _, _, h = DATA[d]
    bcol = COLORS[BAR_KEYS[i]]
    hw, (hlo, hhi) = h
    axB.bar(i, hw, width=0.52, facecolor=bcol, alpha=0.24, edgecolor=bcol, linewidth=1.4, zorder=2)
    axB.errorbar(i, hw, yerr=[[hw - hlo], [hhi - hw]], fmt="none", ecolor=bcol, elinewidth=1.4,
                 capsize=3, alpha=0.9, zorder=3)
    axB.text(i, hhi + 0.5, f"{hw:.1f}", color=bcol, fontsize=9, va="bottom", ha="center", fontweight="bold")

# integer ticks every 2 pts (7-15) with a little headroom so no CI cap is clipped
axT.set_ylim(6.0, 16.4); axT.set_yticks([7, 9, 11, 13, 15])
axB.set_ylim(50, 70);    axB.set_yticks([50, 55, 60, 65, 70])
axT.set_xlim(-0.6, len(DIVS) - 0.4)
axT.set_ylabel("Qwen 1.7B vs gpt-4\nWin Rate (%)", fontsize=11)
axB.set_ylabel("Canonical vs Amari\nWin Rate (%)", fontsize=11)
axB.set_xticks(x); axB.set_xticklabels(DIVS, fontsize=12)
plt.setp(axT.get_xticklabels(), visible=False)
axT.tick_params(axis="x", length=0)
for ax in (axT, axB):
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)

legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CANON, ms=9,
                 label="Canonical (Ours)  $f'(1)=f''(1)$"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AMARI, ms=9,
                 label="Amari  $f'(1)=0$")]
axT.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=9.5,
           frameon=True, columnspacing=1.6, handletextpad=0.5)
fig.suptitle("Arena-Hard Results", fontsize=13.5, y=0.985)

out = "results/stageB_normcmp_wr_h2h"
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
print(f"[saved] {out}.png/.pdf")
