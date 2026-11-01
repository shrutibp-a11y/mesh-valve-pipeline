"""Quantitative mesh quality for every structure of every patient.

Per mesh: closed (watertight), consistent winding, boundary edges, non-manifold edges, connected
components, degenerate faces (< 1e-6 mm2), triangle minimum angle, edge length, self-intersection
(Open3D) and, for the LV, the volume enclosed after capping the planar basal boundary with a
triangle fan, compared with the label-derived EDV (metrics.json).

usage: python revision/05_mesh_quality.py [outputs_dir] [results_dir] [--skip-selfintersection] [--limit N]
(the self-intersection test takes ~15-30 s per patient on one CPU core)
"""
import sys, os, argparse, time, json, numpy as np, pandas as pd, trimesh, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annulus_estimators import boundary_loop
ap = argparse.ArgumentParser()
ap.add_argument("outputs", nargs="?", default="member1/outputs"); ap.add_argument("results", nargs="?", default="revision/results")
ap.add_argument("--skip-selfintersection", action="store_true"); ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args(); os.makedirs(args.results, exist_ok=True)
if not args.skip_selfintersection:
    import open3d as o3d

def cap_and_volume(m):
    idx = boundary_loop(m, return_indices=True)
    if idx is None:
        return (m.volume if m.is_watertight else np.nan), False
    v = np.asarray(m.vertices); c = v[idx].mean(0)
    V = np.vstack([v, c]); ci = len(v)
    F = [[idx[i], idx[(i + 1) % len(idx)], ci] for i in range(len(idx))]
    capped = trimesh.Trimesh(V, np.vstack([m.faces, np.array(F)]), process=False)
    trimesh.repair.fix_normals(capped)
    return (abs(capped.volume) if capped.is_watertight else np.nan), True

def quality(m):
    A = m.area_faces; mina = np.degrees(m.face_angles).min(1)
    u, c = np.unique(m.edges_sorted, axis=0, return_counts=True)
    q = dict(n_vert=len(m.vertices), n_face=len(m.faces), watertight=bool(m.is_watertight), winding_ok=bool(m.is_winding_consistent),
             n_boundary_edges=int((c == 1).sum()), n_nonmanifold_edges=int((c > 2).sum()), components=int(m.body_count),
             degenerate_faces=int((A < 1e-6).sum()), min_angle_med=float(np.median(mina)), min_angle_p5=float(np.percentile(mina, 5)),
             frac_minangle_lt10=float((mina < 10).mean()), edge_mean=float(m.edges_unique_length.mean()), edge_sd=float(m.edges_unique_length.std()),
             surface_area_mm2=float(A.sum()))
    if args.skip_selfintersection:
        q["self_intersecting"] = np.nan
    else:
        t = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(m.vertices)), o3d.utility.Vector3iVector(np.asarray(m.faces)))
        q["self_intersecting"] = bool(t.is_self_intersecting())
    return q

pids = sorted(p for p in os.listdir(args.outputs) if p.startswith("patient"))[: args.limit]
csv_path = os.path.join(args.results, "mesh_quality.csv")
rows = []; t0 = time.time()
if os.path.exists(csv_path):          # resume an interrupted run
    prev = pd.read_csv(csv_path); full = prev.groupby("patient").size()
    done = set(full[full == 4].index); rows = prev[prev.patient.isin(done)].to_dict("records")
    pids = [p for p in pids if p not in done]; print("resuming:", len(done), "patients already done", flush=True)
for i, pid in enumerate(pids):
    for name, path in [("LV", "LV.stl"), ("MYO", "MYO.stl"), ("RV", "RV.stl"), ("valve", "mitral_valve.obj")]:
        m = trimesh.load(os.path.join(args.outputs, pid, path), process=True)
        q = quality(m); q.update(patient=pid, structure=name)
        if name == "LV":
            vol, capped = cap_and_volume(m); q["mesh_volume_mL"] = vol / 1000 if vol == vol else np.nan; q["capped"] = capped
        rows.append(q)
    if i % 5 == 4:
        pd.DataFrame(rows).to_csv(os.path.join(args.results, "mesh_quality.csv"), index=False)
        print(i + 1, "patients done, %.0fs" % (time.time() - t0), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(args.results, "mesh_quality.csv"), index=False); print("DONE", flush=True)
