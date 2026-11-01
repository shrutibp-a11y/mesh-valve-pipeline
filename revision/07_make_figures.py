"""Regenerate Fig. 2 (annulus ellipses and valve position for a representative normal case) and Fig. 3 (area vs EDV).
usage (from the repository root): python revision/07_make_figures.py [outputs_dir] [results_dir]"""
import sys, os, io, contextlib, numpy as np, pandas as pd, trimesh, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annulus_estimators import basal_vertices, two_sigma_axes, fit_ellipse_lsq_full
from member1.valve_modeling import model_mitral_valve
OUT = sys.argv[1] if len(sys.argv) > 1 else "member1/outputs"
RES = sys.argv[2] if len(sys.argv) > 2 else "revision/results"
os.makedirs(os.path.join(RES, "figs"), exist_ok=True); tmp = os.path.join(RES, "_tmp_valve"); os.makedirs(tmp, exist_ok=True)
est = pd.read_csv(os.path.join(RES, "estimates.csv"))

# ---- Fig. 2: representative NOR case = patient whose base-end equivalent diameter is closest to the NOR median
from annulus_estimators import orient_base_high
nor = est[est.group == "NOR"].copy(); nor["eqd"] = 2 * np.sqrt(nor.a2 * nor.b2)
pid = nor.iloc[(nor.eqd - nor.eqd.median()).abs().argsort().iloc[0]].patient
lv = trimesh.load(os.path.join(OUT, pid, "LV.stl")); v0 = np.asarray(lv.vertices)
vo, end = orient_base_high(v0)                      # base at high Z for the XY analysis (Z mirrored if needed)
base, _ = basal_vertices(vo); a1, b1 = two_sigma_axes(base); cx, cy, a2, b2, th = fit_ellipse_lsq_full(base[:, :2])
xy = base[:, :2]; c1 = xy.mean(0); w, vec = np.linalg.eigh(np.cov((xy - c1).T)); vec = vec[:, np.argsort(w)[::-1]]
t = np.linspace(0, 2 * np.pi, 300)
E1 = c1[:, None] + vec @ np.vstack([a1 * np.cos(t), b1 * np.sin(t)])
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
E2 = np.array([[cx], [cy]]) + R @ np.vstack([a2 * np.cos(t), b2 * np.sin(t)])
with contextlib.redirect_stdout(io.StringIO()):
    model_mitral_valve(lv, tmp + "/first", annulus_method="2sigma", base_end="high") if os.makedirs(tmp + "/first", exist_ok=True) is None else None
    model_mitral_valve(lv, tmp + "/rev")  if os.makedirs(tmp + "/rev", exist_ok=True) is None else None
Vf = np.asarray(trimesh.load(os.path.join(tmp, "first", "mitral_valve.obj"), process=False).vertices)
Vr = np.asarray(trimesh.load(os.path.join(tmp, "rev", "mitral_valve.obj"), process=False).vertices)
plt.rcParams.update({"font.size": 15})
fig = plt.figure(figsize=(13, 6.8), dpi=170)
ax = fig.add_axes([0.07, 0.30, 0.38, 0.67])
ax.scatter(base[:, 0], base[:, 1], s=14, c="#9a9a9a", label="Basal vertices (extreme 5% of Z)", zorder=1)
ax.plot(E1[0], E1[1], "--", color="#c0392b", lw=3, label=f"2\u03c3 heuristic (a = {a1:.1f}, b = {b1:.1f} mm)", zorder=2)
ax.plot(E2[0], E2[1], "-", color="#1a7f4b", lw=3.2, label=f"Least-squares fit (a = {a2:.1f}, b = {b2:.1f} mm)", zorder=3)
ax.scatter([cx], [cy], marker="+", s=250, c="k", lw=2.5, zorder=4); ax.set_aspect("equal")
ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.tick_params(labelsize=13)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=12.5, frameon=False)
bx = fig.add_axes([0.56, 0.30, 0.42, 0.67])
bx.scatter(v0[:, 0], v0[:, 2], s=1.2, c="#b5b5b5", rasterized=True, label="LV mesh vertices")
bx.scatter(Vf[:, 0], Vf[:, 2], s=9, c="#c0392b", label="Valve, first release (highest Z)", zorder=3)
bx.scatter(Vr[:, 0], Vr[:, 2], s=9, c="#1a7f4b", label="Valve, revised (detected base)", zorder=3)
bx.set_xlabel("X (mm)"); bx.set_ylabel("Z (mm)"); bx.set_aspect("equal"); bx.tick_params(labelsize=13)
bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=12.5, frameon=False, markerscale=2)
fig.text(0.26, 0.015, "(a)", ha="center", fontsize=18, fontweight="bold"); fig.text(0.77, 0.015, "(b)", ha="center", fontsize=18, fontweight="bold")
plt.savefig(os.path.join(RES, "figs", "fig_valve.png"), dpi=170, facecolor="white"); plt.close()

# ---- Fig. 3: area (least squares) vs EDV, group-wise and pooled fits
from scipy import stats
col = {"NOR": "#3aa06a", "DCM": "#c0392b", "HCM": "#d4a017", "MINF": "#e08030", "ARV": "#3a6fb0"}
assoc = pd.read_csv(os.path.join(RES, "associations.csv"))
rp = assoc[(assoc.y == "EDV") & (assoc.subset == "pooled")].r.iloc[0]; ra = assoc[(assoc.y == "EDV") & (assoc.subset == "group_adjusted")].r.iloc[0]
plt.rcParams.update({"font.size": 15}); fig, ax = plt.subplots(figsize=(10.5, 8), dpi=170)
for g in ["NOR", "DCM", "HCM", "MINF", "ARV"]:
    s = est[est.group == g]; ax.scatter(s.EDV, s.area2, c=col[g], s=70, alpha=.85, edgecolors="white", linewidths=.6, label=g, zorder=3)
    k, b = np.polyfit(s.EDV, s.area2, 1); xs = np.linspace(s.EDV.min(), s.EDV.max(), 20); ax.plot(xs, k * xs + b, color=col[g], lw=2.6, zorder=2)
k, b = np.polyfit(est.EDV, est.area2, 1); xs = np.linspace(est.EDV.min() - 8, est.EDV.max() + 8, 50)
ax.plot(xs, k * xs + b, "--", color="#222222", lw=2.6, label="Pooled fit", zorder=1)
ax.set_xlabel("End-diastolic volume, EDV (mL)", fontsize=17); ax.set_ylabel("Annulus area, least-squares fit (mm$^2$)", fontsize=17)
ax.tick_params(labelsize=15); ax.grid(alpha=.25); ax.legend(loc="upper left", fontsize=14, ncol=2, title="Group-wise fits (solid)", title_fontsize=14)
ax.text(0.98, 0.04, f"pooled r = {rp:.2f}\ngroup-adjusted r = {ra:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=15, bbox=dict(boxstyle="round", fc="white", ec="#888888"))
plt.tight_layout(); plt.savefig(os.path.join(RES, "figs", "fig_corr.png"), dpi=170, facecolor="white")
print("figures written to", os.path.join(RES, "figs"), "(representative case:", pid + ")")
