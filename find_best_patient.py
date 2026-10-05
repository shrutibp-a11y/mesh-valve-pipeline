import os
import json
import glob
import trimesh
import numpy as np

OUTPUTS_DIR = "member1/outputs"

def score_patient(patient_dir):
    """
    Score a patient folder. Higher = better quality.
    Returns (score, report_dict)
    """
    report = {
        "patient":    os.path.basename(patient_dir),
        "folder":     patient_dir,
        "issues":     [],
        "score":      0,
    }

    # ── Check all files exist ─────────────────────────────────────────────
    required = ["full_heart.obj", "LV.stl", "MYO.stl", "RV.stl",
                "mitral_valve.obj", "metrics.json", "region_map.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(patient_dir, f))]
    if missing:
        report["issues"].append(f"Missing files: {missing}")
        report["score"] = -999
        return report

    score = 0

    # ── Metrics score ─────────────────────────────────────────────────────
    try:
        with open(os.path.join(patient_dir, "metrics.json")) as f:
            metrics = json.load(f)

        ef   = metrics["cardiac_function"]["EF_pct"]
        edv  = metrics["cardiac_function"]["EDV_mL"]
        esv  = metrics["cardiac_function"]["ESV_mL"]
        spacing = metrics["patient_info"]["voxel_spacing_mm"]

        report["EF_pct"]  = ef
        report["EDV_mL"]  = edv
        report["ESV_mL"]  = esv
        report["spacing"] = spacing
        report["EF_status"] = metrics["cardiac_function"]["EF_status"]

        # EF score: reward closer to normal (55-70%)
        ef_distance = abs(ef - 62.5)   # distance from middle of normal range
        ef_score = max(0, 40 - ef_distance)  # max 40 points
        score += ef_score

        # Z spacing score: reward smaller Z spacing (better resolution)
        z_spacing = spacing[2] if len(spacing) > 2 else 10.0
        spacing_score = max(0, 20 - z_spacing)  # max 20 points (0mm spacing = 20pts)
        score += spacing_score
        report["z_spacing_mm"] = z_spacing

    except Exception as e:
        report["issues"].append(f"metrics.json error: {e}")
        score -= 20

    # ── Mesh quality score ────────────────────────────────────────────────
    for mesh_file, weight in [("LV.stl", 15), ("MYO.stl", 10), ("RV.stl", 10)]:
        path = os.path.join(patient_dir, mesh_file)
        try:
            mesh = trimesh.load(path)

            # Vertex count — more = more detailed
            vert_score = min(weight, len(mesh.vertices) / 1000)
            score += vert_score

            # Connectivity — fewer bodies = cleaner mesh
            components = trimesh.graph.connected_components(mesh.edges)
            n_components = len(list(components))
            if n_components == 1:
                score += weight   # perfect — single connected mesh
            elif n_components <= 3:
                score += weight * 0.5
            else:
                report["issues"].append(f"{mesh_file}: {n_components} disconnected parts")
                score += weight * 0.1

            report[f"{mesh_file}_vertices"] = len(mesh.vertices)
            report[f"{mesh_file}_components"] = n_components

        except Exception as e:
            report["issues"].append(f"{mesh_file} load error: {e}")
            score -= 10

    # ── Valve score ───────────────────────────────────────────────────────
    valve_path = os.path.join(patient_dir, "mitral_valve.obj")
    try:
        valve = trimesh.load(valve_path)
        if len(valve.vertices) > 100:
            score += 5
        report["valve_vertices"] = len(valve.vertices)
    except Exception as e:
        report["issues"].append(f"valve error: {e}")

    report["score"] = round(score, 1)
    return report


def find_best_patient():
    print("=" * 60)
    print("  Best Patient Selector")
    print("=" * 60)

    # Find all patient folders
    patient_dirs = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "patient*")))

    # Also check root outputs dir (single patient run)
    root_has_files = os.path.exists(os.path.join(OUTPUTS_DIR, "full_heart.obj"))
    if root_has_files and not patient_dirs:
        patient_dirs = [OUTPUTS_DIR]

    if not patient_dirs:
        print(f"\nNo patient folders found in {OUTPUTS_DIR}")
        print("Run: python run_all.py --all")
        return

    print(f"\nScanning {len(patient_dirs)} patient(s)...\n")

    # Score all patients
    reports = []
    for d in patient_dirs:
        r = score_patient(d)
        reports.append(r)

    # Sort by score descending
    reports.sort(key=lambda x: x["score"], reverse=True)

    # Print results table
    print(f"{'Patient':<12} {'Score':>6} {'EF%':>6} {'EDV':>7} {'Z-mm':>6} {'LV verts':>9} {'Issues'}")
    print("-" * 75)

    for r in reports:
        ef      = r.get("EF_pct",   "N/A")
        edv     = r.get("EDV_mL",   "N/A")
        z       = r.get("z_spacing_mm", "N/A")
        lv_v    = r.get("LV.stl_vertices", "N/A")
        issues  = len(r.get("issues", []))
        ef_str  = f"{ef:.1f}" if isinstance(ef, float) else ef
        edv_str = f"{edv:.0f}" if isinstance(edv, float) else edv
        z_str   = f"{z:.1f}"  if isinstance(z,  float) else z
        lv_str  = f"{lv_v:,}" if isinstance(lv_v, int) else lv_v

        marker = " ← BEST" if r == reports[0] else ""
        print(f"{r['patient']:<12} {r['score']:>6.1f} {ef_str:>6} {edv_str:>7} {z_str:>6} {lv_str:>9}  {issues} issue(s){marker}")

    # Best patient summary
    best = reports[0]
    print(f"\n{'=' * 60}")
    print(f"  BEST PATIENT: {best['patient']}")
    print(f"{'=' * 60}")
    print(f"  Score         : {best['score']}")
    print(f"  EF            : {best.get('EF_pct', 'N/A')}%  ({best.get('EF_status', '')})")
    print(f"  EDV           : {best.get('EDV_mL', 'N/A')} mL")
    print(f"  ESV           : {best.get('ESV_mL', 'N/A')} mL")
    print(f"  Z spacing     : {best.get('z_spacing_mm', 'N/A')} mm")
    print(f"  LV vertices   : {best.get('LV.stl_vertices', 'N/A'):,}" if isinstance(best.get('LV.stl_vertices'), int) else f"  LV vertices   : N/A")
    print(f"  Folder        : {best['folder']}")

    if best.get("issues"):
        print(f"\n  Issues:")
        for issue in best["issues"]:
            print(f"    - {issue}")
    else:
        print(f"\n  No issues found — clean output!")

    # Copy best patient files to a "best" folder
    import shutil
    best_dir = os.path.join(OUTPUTS_DIR, "BEST_PATIENT")
    os.makedirs(best_dir, exist_ok=True)
    for fname in ["full_heart.obj", "LV.stl", "MYO.stl", "RV.stl",
                  "mitral_valve.obj", "metrics.json", "region_map.json", "mesh_preview.png"]:
        src = os.path.join(best["folder"], fname)
        dst = os.path.join(best_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Save a summary JSON
    summary = {
        "best_patient":  best["patient"],
        "score":         best["score"],
        "EF_pct":        best.get("EF_pct"),
        "EDV_mL":        best.get("EDV_mL"),
        "ESV_mL":        best.get("ESV_mL"),
        "z_spacing_mm":  best.get("z_spacing_mm"),
        "source_folder": best["folder"],
        "all_scores":    [{
            "patient": r["patient"],
            "score":   r["score"],
            "EF_pct":  r.get("EF_pct"),
            "issues":  r.get("issues", [])
        } for r in reports]
    }
    with open(os.path.join(best_dir, "selection_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Best patient files copied to: {best_dir}/")
    print(f"  Use these files for your final demo.\n")

if __name__ == "__main__":
    find_best_patient()
