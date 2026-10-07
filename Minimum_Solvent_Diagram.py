import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern
from matplotlib.lines import Line2D
from Hunter_Nash import Hunter_Nash
from Hunter_Nash import x_L0, x_LN, x_VN1, Curve_I, Curve_II, Mr, L0, N_frac, S_F
from Minimum_Solvent import Minimum_Solvent
from Minimum_Solvent import k_first, k_last, k_pinch, Delta_min, x_V1_min, M_min, S_F_min

data_I = pd.read_csv("Phase_1_data.csv").to_numpy()[1:]
data_II = pd.read_csv("Phase_2_data.csv").to_numpy()[1:]

data_I = data_I * 100
data_II = data_II * 100

benz_I, acet_I, wat_I = data_I[:, 0], data_I[:, 1], data_I[:, 2]
benz_II, acet_II, wat_II = data_II[:, 0], data_II[:, 1], data_II[:, 2]

# second figure: number of stages against the solvent to feed ratio (it goes to infinity at the minimum), False leaves it out
plot_stages = True

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


def planar_xy(comp):
    comp = np.atleast_2d(comp)[0]
    acet = comp[1] * 100
    wat = comp[2] * 100
    return wat + acet / 2, acet * np.sqrt(3) / 2


def planar_xy_all(comp):
    comp = np.atleast_2d(comp)
    acet = comp[:, 1] * 100
    wat = comp[:, 2] * 100
    return wat + acet / 2, acet * np.sqrt(3) / 2


def annotate(ax, comp, text, dx, dy, color="black", fontsize=15):
    px, py = planar_xy(comp)
    t, l, r = to_ternary(comp)
    at_t, at_l, at_r = planar_to_ternary(px + dx, py + dy)
    ax.plot([t[0], at_t], [l[0], at_l], [r[0], at_r], color="0.35", lw=0.8, zorder=11, clip_on=False)
    ax.text(at_t, at_l, at_r, text, color=color, fontsize=fontsize, ha="center", va="center", zorder=13,
            fontweight="bold", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", lw=0.6, alpha=0.95))


# pinch point construction
fig = plt.figure(figsize=(24, 11))
ax = fig.add_axes([0.02, 0.1, 0.38, 0.82], projection="ternary", ternary_sum=100.0)

ax.set_tlabel("Acetone", fontsize=18)
ax.set_llabel("Benzene", fontsize=18)
ax.set_rlabel("Water", fontsize=18)
ax.tick_params(labelsize=16)
ax.grid()

#phase diagram
ax.plot(Acet_curve, Benz_curve, Wat_curve, color="tab:blue", lw=2.2, zorder=5)

#plait point
ax.scatter(acet_I[-1], benz_I[-1], wat_I[-1], marker="x", color="red", s=90, linewidths=2.5, zorder=9)

#tie lines
tie_lines = np.linspace(0, len(acet_I) - 1, 25, dtype=int)
for i in tie_lines:
    ax.plot([acet_I[i], acet_II[i]], [benz_I[i], benz_II[i]], [wat_I[i], wat_II[i]],
            color="grey", lw=0.8, zorder=2)
ax.plot([0, 0], [100, 0], [0, 100], color="grey", lw=0.8, zorder=2)

#overall mass balance of the minimum solvent rate
MB_F_COLOR = "tab:red"
MB_P_COLOR = "darkcyan"
stage_line(ax, x_VN1, x_L0, color=MB_F_COLOR, lw=2.2, ls="-.", zorder=6)
stage_line(ax, x_LN, x_V1_min, color=MB_P_COLOR, lw=2.2, ls="-.", zorder=6)

stage_line(ax, x_VN1, Delta_min, color="tab:purple", lw=1.4, ls="--", zorder=7, clip_on=False)

k_ext = np.unique(np.append(np.linspace(k_first, k_last, 14, dtype=int), k_pinch))
b_ext = Minimum_Solvent.pinch_positions(Curve_I[k_ext], Curve_II[k_ext], x_VN1, x_LN)
for k, b in zip(k_ext, b_ext):
    P = x_VN1 + b * (x_LN - x_VN1)
    stage_line(ax, Curve_II[k], P, color="tab:purple", lw=1.0, ls=":", zorder=6, clip_on=False)
    ax.scatter(*to_ternary(P), marker="o", color="tab:purple", s=22, zorder=8, clip_on=False)

#pinch tie line, with its extension to the difference point
stage_line(ax, Curve_I[k_pinch], Curve_II[k_pinch], color="black", lw=2.4, zorder=8)
stage_line(ax, Curve_II[k_pinch], Delta_min, color="black", lw=1.6, ls="--", zorder=8, clip_on=False)

#operating line V1 - L0 - Delta (the first operating line of the cascade)
stage_line(ax, x_V1_min, Delta_min, color="tab:purple", lw=1.4, ls="--", zorder=7, clip_on=False)

#difference point
ax.scatter(*to_ternary(Delta_min), marker="o", color="tab:purple", edgecolors="black", s=260, linewidths=1.5,
           zorder=12, clip_on=False)

#feed, solvent, mixing point, raffinate target and extract of the minimum solvent rate
ax.scatter(*to_ternary(x_L0), marker="D", color="white", edgecolors="black", s=100, linewidths=1.3, zorder=10, clip_on=False)
ax.scatter(*to_ternary(x_VN1), marker="v", color="white", edgecolors="black", s=150, linewidths=1.3, zorder=12, clip_on=False)
ax.scatter(*to_ternary(M_min), marker="P", color="white", edgecolors="black", s=120, linewidths=1.3, zorder=10, clip_on=False)
ax.scatter(*to_ternary(x_LN), marker="*", color="white", edgecolors="black", s=200, linewidths=1.2, zorder=10, clip_on=False)
ax.scatter(*to_ternary(x_V1_min), marker="o", color="tab:orange", edgecolors="black", s=110, linewidths=1.2, zorder=10, clip_on=False)

planar_x, planar_y = planar_xy(Delta_min)
ax.text(*planar_to_ternary(planar_x, planar_y + 8), "point of difference", fontsize=16, ha="center", va="center",
        clip_on=False, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
annotate(ax, x_VN1, "S (solvent)", 2, -15)
annotate(ax, x_L0, "L$_0$ (feed)", 8, 8)
annotate(ax, M_min, "M$_{min}$", -8, 10)
annotate(ax, x_LN, "L$_N$ (raffinate)", 0, -15)
annotate(ax, x_V1_min, "V$_1$ (extract)", -14, 6)


#frame around the region that is enlarged below
x_zoom = (98.0, 123.0)
y_zoom = (-1.2, 4.2)
frame_x = [x_zoom[0], x_zoom[1], x_zoom[1], x_zoom[0], x_zoom[0]]
frame_y = [y_zoom[0], y_zoom[0], y_zoom[1], y_zoom[1], y_zoom[0]]
ax.plot(*planar_to_ternary(np.array(frame_x), np.array(frame_y)), color="0.15", lw=1.0, ls=(0, (5, 3)), zorder=1, clip_on=False)

#enlargement of pinch point
ax_z = fig.add_axes([0.57, 0.08, 0.41, 0.3])
ax_z.set_aspect("equal")
ax_z.set_xlim(*x_zoom)
ax_z.set_ylim(*y_zoom)
ax_z.set_xticks([])
ax_z.set_yticks([])

#edges of the triangle and the raffinate side of the phase diagram
ax_z.plot([0, 100], [0, 0], color="black", lw=1.0, zorder=3)
ax_z.plot([100, 50], [0, 100 * np.sqrt(3) / 2], color="black", lw=1.0, zorder=3)
xz, yz = planar_xy_all(Curve_II)
ax_z.plot(xz, yz, color="tab:blue", lw=2.2, zorder=5)
for i in tie_lines:
    xt, yt = planar_xy_all(np.array([[benz_I[i], acet_I[i], wat_I[i]], [benz_II[i], acet_II[i], wat_II[i]]]) / 100)
    ax_z.plot(xt, yt, color="grey", lw=0.8, zorder=2)

#line O-L, extended tie lines and the points P
xs, ys = planar_xy_all(np.array([x_VN1, Delta_min]))
ax_z.plot(xs, ys, color="tab:purple", lw=1.4, ls="--", zorder=7)
for k, b in zip(k_ext, b_ext):
    P = x_VN1 + b * (x_LN - x_VN1)
    xt, yt = planar_xy_all(np.array([Curve_II[k], P]))
    ax_z.plot(xt, yt, color="tab:purple", lw=1.0, ls=":", zorder=6)
    ax_z.scatter(xt[1], yt[1], marker="o", color="tab:purple", s=22, zorder=8)

#pinch tie line and its extension
xt, yt = planar_xy_all(np.array([Curve_I[k_pinch], Curve_II[k_pinch], Delta_min]))
ax_z.plot(xt[0:2], yt[0:2], color="black", lw=2.4, zorder=8)
ax_z.plot(xt[1:3], yt[1:3], color="black", lw=1.6, ls="--", zorder=8)

xn, yn = planar_xy_all(x_LN)
ax_z.scatter(xn, yn, marker="*", color="white", edgecolors="black", s=200, linewidths=1.2, zorder=10)
xd, yd = planar_xy_all(Delta_min)
ax_z.scatter(xd, yd, marker="o", color="tab:purple", edgecolors="black", s=260, linewidths=1.5, zorder=12)
ax_z.text(xd[0], yd[0] + 1.6, "point of difference", fontsize=14, ha="center", va="center",
          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
ax_z.text(xn[0] - 0.4, yn[0] - 0.55, "L$_N$", fontsize=14, ha="right", va="center", fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", lw=0.6, alpha=0.95))

handles = [
    # equilibrium data
    Line2D([], [], color="tab:blue", lw=2.2, label="UNIQUAC binodal curve"),
    Line2D([], [], color="red", marker="x", ls="none", ms=10, mew=2.5, label="plait point"),
    Line2D([], [], color="grey", lw=0.8, label="tie-lines"),
    # streams
    Line2D([], [], color="white", mec="black", marker="D", ls="none", ms=9, label="L$_0$ (feed)"),
    Line2D([], [], color="white", mec="black", marker="v", ls="none", ms=10, label="solvent S = V$_{N+1}$"),
    Line2D([], [], color="white", mec="black", marker="P", ls="none", ms=10, label="mixing point M$_{min}$"),
    Line2D([], [], color="white", mec="black", marker="*", ls="none", ms=14, label="raffinate L$_N$ (1 wt% acetone)"),
    Line2D([], [], color="tab:orange", mec="black", marker="o", ls="none", ms=10, label="extract V$_1$ (minimum solvent)"),
    # construction lines
    Line2D([], [], color=MB_F_COLOR, lw=2.2, ls="-.", label="mass balance line S - M$_{min}$ - L$_0$"),
    Line2D([], [], color=MB_P_COLOR, lw=2.2, ls="-.", label="mass balance line L$_N$ - M$_{min}$ - V$_1$"),
    Line2D([], [], color="tab:purple", lw=1.4, ls="--", label="line S - L$_N$ and operating line V$_1$ - L$_0$"),
    Line2D([], [], color="tab:purple", lw=1.0, ls=":", marker="o", ms=4, label="extended tie lines, cut S - L$_N$ at P"),
    Line2D([], [], color="black", lw=2.4, label="pinch tie line"),
    Line2D([], [], color="black", lw=1.6, ls="--", label="pinch tie line extended"),
    Line2D([], [], color="tab:purple", mec="black", marker="o", ls="none", ms=14, label="point of difference $\\Delta$ = P$_{min}$"),
]
leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.42, 1.12), frameon=True, fancybox=False,
                edgecolor="0.6", framealpha=1.0, fontsize=14, labelspacing=0.7, borderpad=0.9)
plt.savefig("Minimum_Solvent_Diagram.png", dpi=200, bbox_inches="tight", pad_inches=0.4)

# ---------------- number of stages against the solvent to feed ratio ----------------
if plot_stages:
    S_F_list = S_F_min * np.array([1.003, 1.006, 1.012, 1.025, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
    N_list = np.zeros(len(S_F_list))
    for i in range(0, len(S_F_list)):
        solvent_mass = np.array([100.0 * S_F_list[i], 0.0, 0.0])
        V_N1 = Hunter_Nash.mass_to_mol(solvent_mass, Mr)
        M_flow = L0 + V_N1
        x_V1 = Hunter_Nash.extract_point(x_LN, Hunter_Nash.mol_frac(M_flow), Curve_I)
        V1_flow, LN_flow = Hunter_Nash.lever_flows(M_flow, x_V1, x_LN)
        Vs, Ls, ops, L_flow, V_flow, Delta = Hunter_Nash.stage_steps(x_V1, x_LN, L0 - V1_flow * x_V1, Curve_I, Curve_II, 5000)
        n_ideal, N_list[i] = Hunter_Nash.fractional_stages(Ls, x_LN)

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.plot(S_F_list, N_list, color="tab:blue", lw=2.2, marker="o", ms=6, zorder=5)
    ax2.axvline(S_F_min, color="tab:red", lw=1.8, ls="--", zorder=4)
    ax2.scatter([S_F], [N_frac], marker="D", color="white", edgecolors="black", s=110, linewidths=1.3, zorder=10)
    ax2.set_xlabel("Solvent to feed ratio S/F (mass)", fontsize=18)
    ax2.set_ylabel("Number of equilibrium stages", fontsize=18)
    ax2.set_xlim(0, 3.0)
    ax2.set_ylim(0, 40)
    ax2.tick_params(labelsize=16)
    ax2.grid(color="0.8", lw=0.8)
    handles2 = [
        Line2D([], [], color="tab:blue", lw=2.2, marker="o", ms=6, label="Hunter/Nash stage stepping"),
        Line2D([], [], color="tab:red", lw=1.8, ls="--", label="(S/F)$_{min}$ = %.3f" % S_F_min),
        Line2D([], [], color="white", mec="black", marker="D", ls="none", ms=9, label="S/F = %.1f, N = %.2f" % (S_F, N_frac)),
    ]
    ax2.legend(handles=handles2, loc="upper right", frameon=True, fancybox=False, edgecolor="0.6", framealpha=1.0, fontsize=14)
    plt.savefig("Minimum_Solvent_Stages.png", dpi=200, bbox_inches="tight", pad_inches=0.4)

plt.show()
