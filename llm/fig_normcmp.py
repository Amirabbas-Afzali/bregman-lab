"""fig_normcmp.py — amari-vs-canonical Arena-Hard result per divergence (fL3).

Left axis (dots): Arena-Hard v0.1 win rate vs the fixed baseline gpt-4-0314 — amari (grey, f'(1)=0) vs
canonical (dark red, f'(1)=f''(1)), with 95% bootstrap CIs. Right axis (bars): canonical's direct
head-to-head win rate over amari, coloured with each divergence's shared palette colour; the axis floor is
50% (tie), so a bar's height is canonical's margin over a tie and the bars stay clear of the WR dots.
Judge: gpt-4.1-mini-2025-04-14 (validated ≈ gpt-4.1), 500 Arena-Hard v0.1 prompts.

Divergences are in the canonical REGKEYS order (RKL, α-div, FKL, JS, Hel, χ²), matching the tabular figures.

Run from llm/:  python fig_normcmp.py   ->  ../figure4overleaf/fL3_normcmp_wr_h2h.{png,pdf}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from divergences import COLORS   # per-divergence palette shared across all figures

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "mathtext.fontset": "cm", "axes.linewidth": 0.9})

DIVS = ["RKL", "α-div", "FKL", "JS", "Hel", "χ²"]       # canonical REGKEYS order (kl,adiv,rkl,js,hel,chi2)
BAR_KEYS = ["kl", "adiv", "rkl", "js", "hel", "chi2"]   # DIVS -> divergences.COLORS key (kl=RKL, rkl=FKL)

# A: (amari_wr, (lo,hi) offsets) · B: (canon_wr, (lo,hi) offsets) · C: (canon h2h win%, (lo,hi) absolute CI)
DATA = {
    "RKL":   ((8.4,  (1.1, 1.0)), (14.4, (1.4, 1.3)), (63.7, (60.4, 67.1))),
    "α-div": ((9.7,  (1.1, 1.0)), (13.0, (1.3, 1.4)), (59.8, (56.3, 63.2))),
    "FKL":   ((10.1, (1.2, 1.1)), (13.5, (1.3, 1.1)), (57.6, (53.9, 60.8))),
    "JS":    ((10.3, (1.1, 1.3)), (13.7, (1.3, 1.4)), (58.2, (54.7, 61.6))),
    "Hel":   ((9.8,  (1.2, 1.0)), (12.5, (1.2, 1.1)), (58.1, (54.5, 61.8))),
    "χ²":    ((7.2,  (1.0, 0.9)), (13.5, (1.3, 1.2)), (61.8, (58.3, 65.4))),
}

C_AMARI, C_CANON = "#6b7280", "#b0224b"

fig, axL = plt.subplots(figsize=(8.4, 4.6))
axR = axL.twinx()
x = list(range(len(DIVS)))

# right axis: h2h bars (drawn first, behind), coloured per divergence. Axis floor = 50% (tie), so a bar's
# height is canonical's margin over a tie — keeps them short bottom stubs, clear of the WR dots above.
for i, d in enumerate(DIVS):
    a, c, h = DATA[d]
    bcol = COLORS[BAR_KEYS[i]]
    hw, (hlo, hhi) = h
    axR.bar(i, hw, width=0.62, facecolor=bcol, alpha=0.24, edgecolor=bcol, linewidth=1.4, zorder=2)
    axR.errorbar(i, hw, yerr=[[hw - hlo], [hhi - hw]], fmt="none", ecolor=bcol, elinewidth=1.4,
                 capsize=3, alpha=0.9, zorder=3)
    axR.text(i + 0.34, hw, f"{hw:.0f}%", color=bcol, fontsize=8.5, va="center", ha="left", fontweight="bold")

# left axis: amari vs canonical WR dots (no connector — the pair reads from the colour)
for i, d in enumerate(DIVS):
    a, c, h = DATA[d]
    aw, (alo, ahi) = a
    cw, (clo, chi) = c
    axL.errorbar(i, aw, yerr=[[alo], [ahi]], fmt="o", ms=8, color=C_AMARI, ecolor=C_AMARI,
                 elinewidth=1.3, capsize=3, zorder=5)
    axL.errorbar(i, cw, yerr=[[clo], [chi]], fmt="o", ms=8, color=C_CANON, ecolor=C_CANON,
                 elinewidth=1.3, capsize=3, zorder=6)
    axL.text(i - 0.16, aw, f"{aw:.1f}", color=C_AMARI, fontsize=9, va="center", ha="right")
    axL.text(i - 0.16, cw, f"{cw:.1f}", color=C_CANON, fontsize=9, va="center", ha="right", fontweight="bold")

axL.set_ylim(0, 17)
axR.set_ylim(50, 100)                       # floor = tie; bar height = margin over 50%
axL.set_xlim(-0.6, len(DIVS) - 0.4)
axL.set_xticks(x); axL.set_xticklabels(DIVS, fontsize=12)
axL.set_ylabel("Arena-Hard win rate vs gpt-4-0314 (%)", fontsize=11)
axR.set_ylabel("canonical head-to-head win rate (%)", color="#555", fontsize=11)
axR.tick_params(axis="y", colors="#555")
axL.set_axisbelow(True); axL.grid(axis="y", alpha=0.18)

legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AMARI, ms=9, label="amari  $f'(1)=0$"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CANON, ms=9, label="canonical  $f'(1)=f''(1)$"),
          Patch(facecolor="#c9c9c9", alpha=0.5, edgecolor="#9a9a9a",
                label="canonical h2h win rate (>50% = canonical wins)")]
axL.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, fontsize=8.8,
           frameon=True, columnspacing=1.2, handletextpad=0.5)

fig.tight_layout(rect=(0, 0, 1, 0.98))
out = "../figure4overleaf/fL3_normcmp_wr_h2h"
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
print(f"[saved] {out}.png/.pdf")
