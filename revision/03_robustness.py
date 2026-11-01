"""Robustness of the annulus estimators to image position, orientation and the basal threshold.

Reads the LV meshes of every patient, perturbs them and re-runs estimators E1 (2-sigma, first
release) and E2 (least-squares ellipse). Writes robustness.csv and tilt.csv.

usage: python revision/03_robustness.py [outputs_dir] [results_dir]
"""
import sys, os, glob, numpy as np, pandas as pd, trimesh, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annulus_estimators import estimate_all, rotation_matrix, boundary_loops, orient_base_high

OUT = sys.argv[1] if len(sys.argv) > 1 else "member1/outputs"
RES = sys.argv[2] if len(sys.argv) > 2 else "revision/results"
os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(42)
pids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "patient*")))

def pct(new, old):
    return 100.0 * abs(new - old) / old

rows, tilts = [], []
for pid in pids:
    mesh = trimesh.load(os.path.join(OUT, pid, "LV.stl"))
    v, end = orient_base_high(np.asarray(mesh.vertices))      # base at high Z; perturbations are applied to this frame
    cen = v.mean(0)
    b = estimate_all(v, orient=False)
    rec = dict(patient=pid)
    # translation (random shift, up to 200 mm per axis)
    e = estimate_all(v + rng.uniform(-200, 200, 3), orient=False)
    rec["trans_abs_mm2_E1"] = abs(e["area1"] - b["area1"]); rec["trans_abs_mm2_E2"] = abs(e["area2"] - b["area2"])
    # in-plane rotation about Z (37 deg)
    e = estimate_all((v - cen) @ rotation_matrix("z", 37).T + cen, orient=False)
    rec["rotz_pct_E1"] = pct(e["area1"], b["area1"]); rec["rotz_pct_E2"] = pct(e["area2"], b["area2"])
    # out-of-plane tilt: rotation about X and about Y, averaged
    for deg in (5, 10, 20, 30):
        p1, p2 = [], []
        for ax in ("x", "y"):
            e = estimate_all((v - cen) @ rotation_matrix(ax, deg).T + cen, orient=False)
            p1.append(pct(e["area1"], b["area1"])); p2.append(pct(e["area2"], b["area2"]))
        rec[f"tilt{deg}_pct_E1"] = np.mean(p1); rec[f"tilt{deg}_pct_E2"] = np.mean(p2)
    # basal threshold (percentile of Z), signed change relative to P95
    for p in (90, 98):
        e = estimate_all(v, percentile=p, orient=False)
        rec[f"p{p}_signed_pct_E1"] = 100 * (e["area1"] - b["area1"]) / b["area1"]
        rec[f"p{p}_signed_pct_E2"] = 100 * (e["area2"] - b["area2"]) / b["area2"]
    rec["slab_mm"] = b["slab_mm"]; rec["n_base"] = b["n_base"]
    rows.append(rec)
    # inclination of the LV long axis to the slice normal (base end at high Z after orientation):
    # (i) base-slab centroid -> farthest vertex (apex); (ii) base-slab centroid -> apical-slab centroid
    z = v[:, 2]
    base_c = v[z >= np.percentile(z, 95)].mean(0); apex_c = v[z <= np.percentile(z, 5)].mean(0)
    far = v[np.argmax(np.linalg.norm(v - base_c, axis=1))]
    ang = lambda a: float(np.degrees(np.arccos(abs(a[2]) / np.linalg.norm(a))))
    # is the base end of the mesh open, and is its boundary loop in the slice plane?
    # is the base end open? (any boundary loop lying exactly in the plane of the extreme Z of the base end)
    vm = np.asarray(trimesh.Trimesh(v, mesh.faces, process=False).vertices)
    loops_open = int(any(abs(vm[l][:, 2].mean() - z.max()) < 1e-6 and np.ptp(vm[l][:, 2]) < 1e-6
                         for l in boundary_loops(trimesh.Trimesh(v, mesh.faces, process=False))))
    tilts.append(dict(patient=pid, base_end=end, tilt_base_apex_deg=ang(far - base_c), tilt_slab_deg=ang(apex_c - base_c), base_open_planar=loops_open))
pd.DataFrame(rows).to_csv(os.path.join(RES, "robustness.csv"), index=False)
pd.DataFrame(tilts).to_csv(os.path.join(RES, "tilt.csv"), index=False)
print("wrote robustness.csv (%d patients) and tilt.csv (%d patients)" % (len(rows), len(tilts)))
