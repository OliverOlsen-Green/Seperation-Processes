import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern
from matplotlib.lines import Line2D
from Hunter_Nash import Hunter_Nash
from Hunter_Nash import x_L0, M, x_LN, x_V1, x_VN1, Vs, Ls, ops, Delta, N_frac, n_ideal, S_F

data_I = pd.read_csv("Phase_1_data.csv").to_numpy()[1:]
data_II = pd.read_csv("Phase_2_data.csv").to_numpy()[1:]


data_I = data_I * 100
data_II = data_II * 100

benz_I, acet_I, wat_I = data_I[:, 0], data_I[:, 1], data_I[:, 2]
benz_II, acet_II, wat_II = data_II[:, 0], data_II[:, 1], data_II[:, 2]


stage_labels = True
point_labels = False

Acet_curve = np.concatenate([acet_I, acet_II[::-1]])
Benz_curve = np.concatenate([benz_I, benz_II[::-1]])
Wat_curve = np.concatenate([wat_I, wat_II[::-1]])


def to_ternary(x):

    x = np.atleast_2d(x) * 100
    return x[:, 1], x[:, 0], x[:, 2]


def stage_line(ax, a, b, **kwargs):

    ax.plot(*to_ternary(np.array([a, b])), **kwargs)


def planar_to_ternary(x, y):

    acet = y / (np.sqrt(3) / 2)
    wat = x - acet / 2
    benz = 100.0 - acet - wat
    return acet, benz, wat


def label(ax, comp, text, at, color="black", fontsize=14):

    if color == "black" and not point_labels:
        return
    if color != "black" and not stage_labels:
        return
    t, l, r = to_ternary(comp)
    at_t, at_l, at_r = planar_to_ternary(*at)
    ax.plot([t[0], at_t], [l[0], at_l], [r[0], at_r], color="0.35", lw=0.8, zorder=11, clip_on=False)
    ax.text(at_t, at_l, at_r, text, color=color, fontsize=fontsize, ha="center", va="center", zorder=12,
            fontweight="bold" if color != "black" else None, clip_on=False,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))


def in_window(x, lim):
    x = np.atleast_2d(x)
    if lim is None:
        return x
    tmin, tmax, lmin, lmax, rmin, rmax = lim
    tol = 0.01 * (tmax - tmin)
    keep = []
    for k in range(0, x.shape[0]):
        t, l, r = x[k, 1] * 100, x[k, 0] * 100, x[k, 2] * 100
        if tmin - tol <= t <= tmax + tol and lmin - tol <= l <= lmax + tol and rmin - tol <= r <= rmax + tol:
            keep.append(x[k])
    if len(keep) == 0:
        return np.zeros((0, 3))
    return np.array(keep)


def planar_xy(comp):
    comp = np.atleast_2d(comp)[0]
    acet = comp[1] * 100
    wat = comp[2] * 100
    return wat + acet / 2, acet * np.sqrt(3) / 2


def annotate(ax, comp, text, dx, dy, color="black", fontsize=15):
    # always-on label with a leader line; (dx, dy) is the text offset from the point, in diagram units
    px, py = planar_xy(comp)
    t, l, r = to_ternary(comp)
    at_t, at_l, at_r = planar_to_ternary(px + dx, py + dy)
    ax.plot([t[0], at_t], [l[0], at_l], [r[0], at_r], color="0.35", lw=0.8, zorder=11, clip_on=False)
    ax.text(at_t, at_l, at_r, text, color=color, fontsize=fontsize, ha="center", va="center", zorder=13,
            fontweight="bold", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", lw=0.6, alpha=0.95))


def draw_diagram(ax, size=1.0, lim=None, mb_line=True):
    ax.set_tlabel("Acetone", fontsize=18)
    ax.set_llabel("Benzene", fontsize=18)
    ax.set_rlabel("Water", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.grid()

    #phase diagram
    ax.plot(Acet_curve, Benz_curve, Wat_curve, color="tab:blue", lw=2.2 * size, zorder=5)

    #plait point
    ax.scatter(acet_I[-1], benz_I[-1], wat_I[-1], marker="x", color="red", s=90 * size, linewidths=2.5, zorder=9)

    #tie lines
    tie_lines = np.linspace(0, len(acet_I) - 1, 25, dtype=int)
    for i in tie_lines:
        ax.plot([acet_I[i], acet_II[i]], [benz_I[i], benz_II[i]], [wat_I[i], wat_II[i]],
                color="grey", lw=0.8, zorder=2)
    ax.plot([0, 0], [100, 0], [0, 100], color="grey", lw=0.8, zorder=2)

    #LN - M - V1
    if mb_line:
        stage_line(ax, x_LN, x_V1, color="0.15", lw=1.4 * size, ls=":", zorder=6)

    #operating lines (L_j - V_j+1) pointing toa difference point
    for (x_L, y_V) in ops:
        stage_line(ax, x_L, y_V, color="tab:purple", lw=1.8 * size, ls="--", zorder=7)

    #stage tie lines (V_j - L_j)
    for x_V, x_L in zip(Vs, Ls):
        stage_line(ax, x_V, x_L, color="black", lw=2.4 * size, zorder=8)

    #extract and raffinate of each stage
    V_in = in_window(np.array(Vs), lim)
    L_in = in_window(np.array(Ls), lim)
    if len(V_in) > 0:
        ax.scatter(*to_ternary(V_in), marker="o", color="tab:orange", edgecolors="black", s=110 * size,
                   linewidths=1.2, zorder=10, clip_on=False)
    if len(L_in) > 0:
        ax.scatter(*to_ternary(L_in), marker="s", color="tab:green", edgecolors="black", s=100 * size,
                   linewidths=1.2, zorder=10, clip_on=False)

    #feed, mixing point and raffinate target
    if len(in_window(x_L0, lim)) > 0:
        ax.scatter(*to_ternary(x_L0), marker="D", color="white", edgecolors="black", s=100 * size, linewidths=1.3, zorder=10, clip_on=False)
    if len(in_window(M, lim)) > 0:
        ax.scatter(*to_ternary(M), marker="P", color="white", edgecolors="black", s=120 * size, linewidths=1.3, zorder=10, clip_on=False)
    if len(in_window(x_LN, lim)) > 0:
        ax.scatter(*to_ternary(x_LN), marker="*", color="white", edgecolors="black", s=200 * size, linewidths=1.2, zorder=10, clip_on=False)


V_color = "#b35300"
L_color = "#1b6e1b"

# ---------------- main figure ----------------
fig = plt.figure(figsize=(12.5, 12.5))
ax = fig.add_subplot(projection="ternary", ternary_sum=100.0)
draw_diagram(ax)

label(ax, Vs[0], "V$_1$", (25, 31), V_color)
label(ax, Vs[1], "V$_2$", (13, 10.5), V_color)
label(ax, Vs[2], "V$_3$", (12, 3.9), V_color)
label(ax, Vs[3], "V$_4$", (9, -3.4), V_color)
label(ax, Ls[0], "L$_1$", (112, 12), L_color)
label(ax, Ls[1], "L$_2$", (113.5, 7.5), L_color)
label(ax, Ls[2], "L$_3$", (114.5, 3.5), L_color)
label(ax, Ls[3], "L$_4$", (117, -0.8), L_color)
label(ax, x_L0, "L$_0$ (feed)", (76, 29))
label(ax, M, "M", (60, 24))

handles = [
    Line2D([], [], color="tab:blue", lw=2.2, label="UNIQUAC calculated"),
    Line2D([], [], color="red", marker="x", ls="none", ms=10, mew=2.5, label="plait point"),
    Line2D([], [], color="grey", lw=0.8, label="tie-lines"),
    Line2D([], [], color="black", lw=2.4, label="stage tie-lines"),
    Line2D([], [], color="tab:purple", lw=1.8, ls="--", label="operating lines (via $\\Delta$)"),
    Line2D([], [], color="0.15", lw=1.4, ls=":", label="L$_N$ - M - V$_1$"),
    Line2D([], [], color="tab:orange", mec="black", marker="o", ls="none", ms=10, label="extract stages V$_j$"),
    Line2D([], [], color="tab:green", mec="black", marker="s", ls="none", ms=10, label="raffinate stages L$_j$"),
    Line2D([], [], color="white", mec="black", marker="D", ls="none", ms=9, label="feed L$_0$"),
    Line2D([], [], color="white", mec="black", marker="P", ls="none", ms=10, label="mixing point M"),
    Line2D([], [], color="white", mec="black", marker="*", ls="none", ms=14, label="raffinate L$_N$ (1 wt% acetone)"),
]
ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.2, 0.5), frameon=False, fontsize=14)

# Prevent text cutoff on save
plt.subplots_adjust(left=0.15, right=0.75, bottom=0.15)
plt.savefig("Hunter_Nash_Diagram.png", dpi=300, bbox_inches="tight", pad_inches=0.5)

# ---------------- zoomed figure: the corners of the diagram where the stages are close together ----------------
fig2 = plt.figure(figsize=(20, 18))

ax_b = fig2.add_subplot(2, 2, 1, projection="ternary", ternary_sum=100.0)
draw_diagram(ax_b, 1.2, (0, 12, 88, 100, 0, 12))
ax_b.set_ternary_lim(0, 12, 88, 100, 0, 12)
label(ax_b, Vs[1], "V$_2$", (6.2, 5.3), V_color, 18)
label(ax_b, Vs[2], "V$_3$", (4.6, 3.6), V_color, 18)
label(ax_b, Vs[3], "V$_4$", (3.6, 1.0), V_color, 18)

ax_w = fig2.add_subplot(2, 2, 2, projection="ternary", ternary_sum=100.0)
draw_diagram(ax_w, 1.2, (0, 12, 0, 12, 88, 100))
ax_w.set_ternary_lim(0, 12, 0, 12, 88, 100)
label(ax_w, Ls[0], "L$_1$", (94.3, 7.6), L_color, 18)
label(ax_w, Ls[1], "L$_2$", (96.3, 3.0), L_color, 18)
label(ax_w, Ls[2], "L$_3$", (97.4, 1.15), L_color, 18)
label(ax_w, Ls[3], "L$_4$", (102.6, 2.9), L_color, 18)
label(ax_w, x_LN, "L$_N$", (102.6, 1.4), "black", 18)

ax_b2 = fig2.add_subplot(2, 2, 3, projection="ternary", ternary_sum=100.0)
draw_diagram(ax_b2, 1.6, (0, 3, 97, 100, 0, 3))
ax_b2.set_ternary_lim(0, 3, 97, 100, 0, 3)
label(ax_b2, Vs[2], "V$_3$", (1.9, 1.5), V_color, 18)
label(ax_b2, Vs[3], "V$_4$", (1.4, 0.65), V_color, 18)

ax_w2 = fig2.add_subplot(2, 2, 4, projection="ternary", ternary_sum=100.0)
draw_diagram(ax_w2, 1.6, (0, 3, 0, 3, 97, 100))
ax_w2.set_ternary_lim(0, 3, 0, 3, 97, 100)
label(ax_w2, Ls[1], "L$_2$", (98.45, 1.95), L_color, 18)
label(ax_w2, Ls[2], "L$_3$", (99.0, 1.05), L_color, 18)
label(ax_w2, Ls[3], "L$_4$", (98.2, 0.72), L_color, 18)
label(ax_w2, x_LN, "L$_N$", (98.5, 0.2), "black", 18)

plt.subplots_adjust(wspace=0.4, hspace=0.35)
plt.savefig("Hunter_Nash_Zoom.png", dpi=150, bbox_inches="tight", pad_inches=0.4)
# ---------------- difference point: the operating lines are extended until they meet in Delta ----------------
fig3 = plt.figure(figsize=(26, 8))
ax_d = fig3.add_axes([0.03, 0.12, 0.25, 0.78], projection="ternary", ternary_sum=100.0)
draw_diagram(ax_d, 0.9, mb_line=False)

#overall mass balance: F + S = M = L_N + V_1. The S-F line and the L_N-V_1 line cross at the mixing point M
MB_F_COLOR = "tab:red"
MB_P_COLOR = "darkcyan"
stage_line(ax_d, x_VN1, x_L0, color=MB_F_COLOR, lw=2.2, ls="-.", zorder=6)
stage_line(ax_d, x_LN, x_V1, color=MB_P_COLOR, lw=2.2, ls="-.", zorder=6)

#operating lines extended beyond the diagram, V_j+1 -> L_j -> Delta (the first one is V1 -> L0 -> Delta)
operating_ends = [x_V1] + [y_V for (x_L, y_V) in ops]
for y_V in operating_ends:
    stage_line(ax_d, y_V, Delta, color="tab:purple", lw=1.4, ls="--", zorder=7, clip_on=False)
#solvent V_N+1 (pure benzene) -> raffinate LN -> Delta, this line is the bottom edge of the diagram
stage_line(ax_d, x_VN1, Delta, color="tab:purple", lw=1.4, ls="--", zorder=7, clip_on=False)

ax_d.scatter(*to_ternary(Delta), marker="o", color="tab:purple", edgecolors="black", s=260, linewidths=1.5,
             zorder=12, clip_on=False)
ax_d.scatter(*to_ternary(x_VN1), marker="v", color="white", edgecolors="black", s=150, linewidths=1.3,
             zorder=12, clip_on=False)

planar_x = Delta[2] * 100 + Delta[1] * 50
planar_y = Delta[1] * 100 * np.sqrt(3) / 2
ax_d.text(*planar_to_ternary(planar_x, planar_y + 8), "point of difference", fontsize=16, ha="center", va="center",
          clip_on=False, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
annotate(ax_d, x_VN1, "S (solvent)", 2, -15)
annotate(ax_d, x_L0, "L$_0$ (feed)", 8, 8)
annotate(ax_d, M, "M", 0, 8)
annotate(ax_d, x_LN, "L$_N$ (raffinate)", 0, -15)
label(ax_d, Vs[0], "V$_1$", (25, 31), V_color, 12)
label(ax_d, Vs[1], "V$_2$", (13, 10.5), V_color, 12)
label(ax_d, Vs[2], "V$_3$", (12, 3.9), V_color, 12)
label(ax_d, Vs[3], "V$_4$", (9, -3.4), V_color, 12)
label(ax_d, Ls[0], "L$_1$", (112, 12), L_color, 12)
label(ax_d, Ls[1], "L$_2$", (113.5, 7.5), L_color, 12)
label(ax_d, Ls[2], "L$_3$", (114.5, 3.5), L_color, 12)
label(ax_d, Ls[3], "L$_4$", (117, -0.8), L_color, 12)

handles_d = [
    # equilibrium data
    Line2D([], [], color="tab:blue", lw=2.2, label="UNIQUAC binodal curve"),
    Line2D([], [], color="red", marker="x", ls="none", ms=10, mew=2.5, label="plait point"),
    Line2D([], [], color="grey", lw=0.8, label="tie-lines"),
    # streams
    Line2D([], [], color="white", mec="black", marker="D", ls="none", ms=9, label="L$_0$ (feed)"),
    Line2D([], [], color="white", mec="black", marker="v", ls="none", ms=10, label="solvent S = V$_{N+1}$"),
    Line2D([], [], color="white", mec="black", marker="P", ls="none", ms=10, label="mixing point M"),
    Line2D([], [], color="white", mec="black", marker="*", ls="none", ms=14, label="raffinate L$_N$ (1 wt% acetone)"),
    Line2D([], [], color="tab:orange", mec="black", marker="o", ls="none", ms=10, label="extract stages V$_j$"),
    Line2D([], [], color="tab:green", mec="black", marker="s", ls="none", ms=10, label="raffinate stages L$_j$"),
    # construction lines
    Line2D([], [], color=MB_F_COLOR, lw=2.2, ls="-.", label="mass balance line S - M - L$_0$"),
    Line2D([], [], color=MB_P_COLOR, lw=2.2, ls="-.", label="mass balance line L$_N$ - M - V$_1$"),
    Line2D([], [], color="black", lw=2.4, label="stage tie-lines"),
    Line2D([], [], color="tab:purple", lw=1.4, ls="--", label="operating lines through $\\Delta$"),
    Line2D([], [], color="tab:purple", mec="black", marker="o", ls="none", ms=14, label="point of difference $\\Delta$"),
]
leg = ax_d.legend(handles=handles_d, loc="upper left", bbox_to_anchor=(2.15, 1.12), frameon=True, fancybox=False,
                  edgecolor="0.6", framealpha=1.0, fontsize=14, labelspacing=0.7, borderpad=0.9)
plt.savefig("Hunter_Nash_Delta.png", dpi=200, bbox_inches="tight", pad_inches=0.4)

plt.show()