"""Run annulus estimators E1 (2-sigma), E2 (least-squares ellipse) and E3 (Feret) on every LV mesh.

Main columns use the base end detected from the mesh geometry. Columns with suffix _top reproduce the
first release, which always used the highest-Z vertices (the apical end in 98 of 100 ACDC cases).
usage: python revision/01_estimate_annulus.py [outputs_dir] [results_dir]"""
import sys, os, numpy as np, pandas as pd, trimesh, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annulus_estimators import estimate_all, base_end
OUT = sys.argv[1] if len(sys.argv) > 1 else "member1/outputs"
RES = sys.argv[2] if len(sys.argv) > 2 else "revision/results"
df = pd.read_csv(os.path.join(RES, "patient_summary.csv"))
rows = []
for pid in df.patient:
    v = np.asarray(trimesh.load(os.path.join(OUT, pid, "LV.stl")).vertices)
    r = estimate_all(v, orient=True); t = estimate_all(v, orient=False)
    _, lo, hi = base_end(v)
    r.update(patient=pid, section_area_low_end=lo, section_area_high_end=hi)
    for k in ("a1", "b1", "area1", "a2", "b2", "area2"): r[k + "_top"] = t[k]
    rows.append(r)
out = df.merge(pd.DataFrame(rows), on="patient")
out.to_csv(os.path.join(RES, "estimates.csv"), index=False)
print("base end: low-Z in %d, high-Z in %d of %d" % ((out.base_end == "low").sum(), (out.base_end == "high").sum(), len(out)))
print("stored metrics.json axis a vs first-release E1 (top slab): max |da| = %.3f mm, max |darea| = %.2f mm2" % ((out.a - out.a1_top).abs().max(), (out.area - out.area1_top).abs().max()))
print("   (small if metrics.json comes from the FIRST release; large is expected after regenerating with the corrected code)")
print("stored metrics.json axis a vs least-squares E2 at the base: max |da| = %.3f mm (about 0.005 mm if regenerated with the corrected code)" % (out.a - out.a2).abs().max())
print("ratio a1/a2 at base: mean %.3f SD %.3f" % ((out.a1 / out.a2).mean(), (out.a1 / out.a2).std()))
