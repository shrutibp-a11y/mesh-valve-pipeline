"""Cross-check of the detected base end with two independent criteria.

V1  thin-slab cross-sections at 10/20/30% from each end (the rule used by the pipeline)
V2  mean XY distance from the vertex centroid axis in the lowest vs highest quartile of Z (wider end = base)
V3  where the mesh is open at both ends, the end with the longer boundary loop (wider end = base)
usage: python revision/08_orientation_check.py [outputs_dir] [results_dir]"""
import sys, os, numpy as np, pandas as pd, trimesh, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annulus_estimators import base_end
OUT = sys.argv[1] if len(sys.argv) > 1 else "member1/outputs"
RES = sys.argv[2] if len(sys.argv) > 2 else "revision/results"
def loops(m):
    u, c = np.unique(m.edges_sorted, axis=0, return_counts=True); b = u[c == 1]; adj = {}
    for a, bb in b: adj.setdefault(int(a), []).append(int(bb)); adj.setdefault(int(bb), []).append(int(a))
    seen = set(); L = []
    for s in list(adj):
        if all((min(s, n), max(s, n)) in seen for n in adj[s]): continue
        lp = [s]; cur = s
        while True:
            nx = [n for n in adj[cur] if (min(cur, n), max(cur, n)) not in seen]
            if not nx: break
            n = nx[0]; seen.add((min(cur, n), max(cur, n)))
            if n == s: break
            lp.append(n); cur = n
        if len(lp) > 2: L.append(lp)
    return L
rows = []
for pid in sorted(p for p in os.listdir(OUT) if p.startswith("patient")):
    m = trimesh.load(os.path.join(OUT, pid, "LV.stl")); V = np.asarray(m.vertices); z = V[:, 2]; z0, z1 = z.min(), z.max()
    v1, lo, hi = base_end(V)
    c = V[:, :2].mean(0); r = np.linalg.norm(V[:, :2] - c, axis=1)
    v2 = "low" if r[z <= np.percentile(z, 25)].mean() > r[z >= np.percentile(z, 75)].mean() else "high"
    L = loops(m)
    nl = sum(len(l) for l in L if abs(V[l][:, 2].mean() - z0) < 1e-6 and np.ptp(V[l][:, 2]) < 1e-6)
    nh = sum(len(l) for l in L if abs(V[l][:, 2].mean() - z1) < 1e-6 and np.ptp(V[l][:, 2]) < 1e-6)
    v3 = ("low" if nl > nh else "high") if (nl > 0 and nh > 0 and nl != nh) else "n/a"
    rows.append(dict(patient=pid, v1_slab=v1, v2_radial=v2, v3_loop=v3, loop_low_edges=nl, loop_high_edges=nh, area_ratio_low_over_high=lo / max(hi, 1e-9)))
D = pd.DataFrame(rows)
D["unanimous"] = (D.v1_slab == D.v2_radial) & ((D.v3_loop == "n/a") | (D.v3_loop == D.v1_slab))
D.to_csv(os.path.join(RES, "orientation_check.csv"), index=False)
print("V1 low:", (D.v1_slab == "low").sum(), "| V2 agrees with V1:", (D.v1_slab == D.v2_radial).sum(),
      "| V3 available:", (D.v3_loop != "n/a").sum(), "agrees:", ((D.v3_loop == D.v1_slab) & (D.v3_loop != "n/a")).sum(),
      "| all criteria agree:", D.unanimous.sum(), "| ambiguous:", D[~D.unanimous].patient.tolist())
print(D[~D.unanimous].to_string(index=False))
