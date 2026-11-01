"""Wall-clock time of the valve-reconstruction stage (LV mesh -> mitral_valve.obj incl. export).
usage: python revision/06_timing_valve_stage.py [outputs_dir] [results_dir]  (run from the repository root)"""
import sys, os, io, time, contextlib, platform, json, numpy as np, pandas as pd, trimesh, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from member1.valve_modeling import model_mitral_valve
OUT = sys.argv[1] if len(sys.argv) > 1 else "member1/outputs"
RES = sys.argv[2] if len(sys.argv) > 2 else "revision/results"
pids = sorted(p for p in os.listdir(OUT) if p.startswith("patient"))
tmp = os.path.join(RES, "_tmp_valve"); os.makedirs(tmp, exist_ok=True)
def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()): return f(*a, **k)
quiet(model_mitral_valve, trimesh.load(os.path.join(OUT, pids[0], "LV.stl")), tmp)   # warm-up, not counted
rows = []
for pid in pids:
    t0 = time.perf_counter(); m = trimesh.load(os.path.join(OUT, pid, "LV.stl")); tl = time.perf_counter() - t0
    t0 = time.perf_counter(); quiet(model_mitral_valve, m, tmp); tv = time.perf_counter() - t0
    rows.append(dict(patient=pid, t_load_s=tl, t_valve_s=tv))
pd.DataFrame(rows).to_csv(os.path.join(RES, "timing_valve_stage.csv"), index=False)
cpu = next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
ram = next((round(int(l.split()[1]) / 2**20, 1) for l in open("/proc/meminfo") if l.startswith("MemTotal")), None)
import numpy, scipy, trimesh
hw = dict(cpu=cpu, logical_cores=os.cpu_count(), ram_gb=ram, os=platform.platform(), python=platform.python_version(),
          numpy=numpy.__version__, scipy=scipy.__version__, trimesh=trimesh.__version__)
json.dump(hw, open(os.path.join(RES, "hardware.json"), "w"), indent=2); print(hw)
T = pd.DataFrame(rows); print("valve stage: mean %.1f ms SD %.1f ms (median %.1f, range %.1f-%.1f)" % (T.t_valve_s.mean()*1e3, T.t_valve_s.std()*1e3, T.t_valve_s.median()*1e3, T.t_valve_s.min()*1e3, T.t_valve_s.max()*1e3))
