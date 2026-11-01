"""Collect per-patient metrics from the pipeline outputs into one table.

ACDC training groups follow the official patient numbering: 001-020 DCM, 021-040 HCM,
041-060 MINF, 061-080 NOR, 081-100 ARV (check against the Info.cfg files of the dataset).

usage: python revision/00_collect_metrics.py [outputs_dir] [results_dir]
"""
import sys, os, glob, json, pandas as pd
OUT = sys.argv[1] if len(sys.argv) > 1 else "member1/outputs"
RES = sys.argv[2] if len(sys.argv) > 2 else "revision/results"
os.makedirs(RES, exist_ok=True)
def group(n):
    return "DCM" if n <= 20 else "HCM" if n <= 40 else "MINF" if n <= 60 else "NOR" if n <= 80 else "ARV"
rows = []
for d in sorted(glob.glob(os.path.join(OUT, "patient*"))):
    pid = os.path.basename(d); m = json.load(open(os.path.join(d, "metrics.json")))
    cf, vg = m["cardiac_function"], m["valve_geometry"]
    rows.append(dict(patient=pid, group=group(int(pid.replace("patient", ""))), EDV=cf["EDV_mL"], ESV=cf["ESV_mL"],
                     EF=cf["EF_pct"], a=vg["semi_major_axis_mm"], b=vg["semi_minor_axis_mm"], area=vg["annulus_area_mm2"],
                     zsp=m["patient_info"]["voxel_spacing_mm"][2], xsp=m["patient_info"]["voxel_spacing_mm"][0]))
pd.DataFrame(rows).to_csv(os.path.join(RES, "patient_summary.csv"), index=False)
print("wrote patient_summary.csv with", len(rows), "patients")
