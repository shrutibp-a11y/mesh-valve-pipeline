"""(i) Volume enclosed by each LV mesh after capping every planar boundary loop with a triangle fan,
compared with the label-derived EDV, and the number of intersecting triangle pairs of the LV mesh.
(ii) Quality of the valve meshes regenerated with the revised code (base end detected, least-squares
ellipse, degenerate triangles removed).

usage (repository root): python revision/05b_volume_and_valve.py [outputs_dir] [results_dir]"""
import sys, os, io, contextlib, numpy as np, pandas as pd, trimesh, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import open3d as o3d
from member1.valve_modeling import model_mitral_valve
OUT = sys.argv[1] if len(sys.argv) > 1 else "member1/outputs"
RES = sys.argv[2] if len(sys.argv) > 2 else "revision/results"
est = pd.read_csv(os.path.join(RES, "estimates.csv")).set_index("patient")
tmp = os.path.join(RES, "_tmp_valve2"); os.makedirs(tmp, exist_ok=True)

def boundary_loops(m):
    u, c = np.unique(m.edges_sorted, axis=0, return_counts=True); b = u[c == 1]
    adj = {}
    for a, bb in b: adj.setdefault(int(a), []).append(int(bb)); adj.setdefault(int(bb), []).append(int(a))
    seen = set(); loops = []
    for s in list(adj):
        if all((min(s, n), max(s, n)) in seen for n in adj[s]): continue
        loop = [s]; cur = s
        while True:
            nx = [n for n in adj[cur] if (min(cur, n), max(cur, n)) not in seen]
            if not nx: break
            n = nx[0]; seen.add((min(cur, n), max(cur, n)))
            if n == s: break
            loop.append(n); cur = n
        if len(loop) > 2: loops.append(loop)
    return loops

def capped_volume(m):
    loops = boundary_loops(m); v = np.asarray(m.vertices); V = v.copy(); F = [np.asarray(m.faces)]
    for lp in loops:
        c = v[lp].mean(0); ci = len(V); V = np.vstack([V, c])
        F.append(np.array([[lp[i], lp[(i + 1) % len(lp)], ci] for i in range(len(lp))]))
    cm = trimesh.Trimesh(V, np.vstack(F), process=False); trimesh.repair.fix_normals(cm)
    return (abs(cm.volume) / 1000 if cm.is_watertight else np.nan), len(loops)

def o3d_mesh(m):
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(m.vertices)), o3d.utility.Vector3iVector(np.asarray(m.faces)))

rows_v, rows_q = [], []
pv, pq = os.path.join(RES, "lv_volume.csv"), os.path.join(RES, "valve_quality_revised.csv")
if os.path.exists(pv) and os.path.exists(pq):          # resume an interrupted run
    rows_v = pd.read_csv(pv).to_dict("records"); rows_q = pd.read_csv(pq).to_dict("records")
    print("resuming:", len(rows_v), "patients already done", flush=True)
done = {r["patient"] for r in rows_v}
for pid in [p for p in est.index if p not in done]:
    m = trimesh.load(os.path.join(OUT, pid, "LV.stl"))
    vol, nloops = capped_volume(m)
    pairs = len(np.asarray(o3d_mesh(m).get_self_intersecting_triangles()))
    rows_v.append(dict(patient=pid, edv_label_mL=est.loc[pid, "EDV"], mesh_volume_mL=vol, n_boundary_loops=nloops, lv_selfint_pairs=pairs, lv_faces=len(m.faces)))
    with contextlib.redirect_stdout(io.StringIO()):
        vm, info = model_mitral_valve(m, tmp)
    vm = trimesh.load(os.path.join(tmp, "mitral_valve.obj"), process=True)
    u, c = np.unique(vm.edges_sorted, axis=0, return_counts=True); mina = np.degrees(vm.face_angles).min(1)
    rows_q.append(dict(patient=pid, structure="valve_revised", n_vert=len(vm.vertices), n_face=len(vm.faces), watertight=bool(vm.is_watertight),
                       n_boundary_edges=int((c == 1).sum()), n_nonmanifold_edges=int((c > 2).sum()), components=int(vm.body_count),
                       degenerate_faces=int((vm.area_faces < 1e-6).sum()), min_angle_med=float(np.median(mina)), min_angle_p5=float(np.percentile(mina, 5)),
                       self_intersecting=bool(o3d_mesh(vm).is_self_intersecting()), base_end=info["base_end"], ring_a=info["semi_major_axis_mm"],
                       ring_b=info["semi_minor_axis_mm"], valve_z_range_mm=float(np.ptp(np.asarray(vm.vertices)[:, 2])), edge_mean=float(vm.edges_unique_length.mean())))
    if len(rows_v) % 10 == 0:
        pd.DataFrame(rows_v).to_csv(os.path.join(RES, "lv_volume.csv"), index=False); pd.DataFrame(rows_q).to_csv(os.path.join(RES, "valve_quality_revised.csv"), index=False)
        print(len(rows_v), "done", flush=True)
pd.DataFrame(rows_v).to_csv(os.path.join(RES, "lv_volume.csv"), index=False); pd.DataFrame(rows_q).to_csv(os.path.join(RES, "valve_quality_revised.csv"), index=False); print("DONE", flush=True)
