import os
import sys
import glob
import json
import argparse
import time
import traceback
import platform
import csv

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from member1.load_acdc     import load_patient, extract_masks
from member1.mesh_pipeline import mask_to_mesh, fuse_meshes, visualize_meshes
from member1.valve_modeling import model_mitral_valve
from member1.compute_metrics import compute_cardiac_metrics, compute_wall_thickness


# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_DIRS = [
    "data/training",
    "training",
    "data",
    "ACDC/training",
]

OUTPUT_BASE = "member1/outputs"


def find_data_dir():
    """Find the ACDC training data directory."""
    for d in DATA_DIRS:
        if os.path.isdir(d):
            patients = glob.glob(os.path.join(d, "patient*"))
            if patients:
                return d
    return None


def run_patient(patient_dir, output_dir, annulus_method="lsq", timings=None):
    """
    Run the full pipeline for one patient.
    Returns True on success, False on failure.
    """
    patient_name = os.path.basename(patient_dir)
    t_start = time.perf_counter()
    stage = {}
    print(f"\n{'='*60}")
    print(f"  Processing: {patient_name}")
    print(f"{'='*60}")

    try:
        # ── Step 3: Load MRI data ──────────────────────────────────────────
        print("\n[Step 3] Loading MRI data...")
        _t = time.perf_counter()
        mri_ed,    labels_ed, spacing = load_patient(patient_dir, frame="ED")
        _,         labels_es, _       = load_patient(patient_dir, frame="ES")

        masks_ed = extract_masks(labels_ed)
        masks_es = extract_masks(labels_es)

        print(f"  Volume shape   : {mri_ed.shape}")
        print(f"  Voxel spacing  : {tuple(round(s,2) for s in spacing)} mm")
        print(f"  LV voxels (ED) : {masks_ed['LV'].sum():,}")
        stage["load_s"] = time.perf_counter() - _t

        # ── Step 4: Generate meshes ───────────────────────────────────────
        print("\n[Step 4] Generating 3D meshes...")
        _t = time.perf_counter()
        lv_mesh  = mask_to_mesh(masks_ed["LV"],  spacing, "LV",  output_dir)
        myo_mesh = mask_to_mesh(masks_ed["MYO"], spacing, "MYO", output_dir)
        rv_mesh  = mask_to_mesh(masks_ed["RV"],  spacing, "RV",  output_dir)

        print("\n  Fusing into full_heart.obj...")
        full_heart = fuse_meshes(lv_mesh, rv_mesh, myo_mesh, output_dir)
        stage["mesh_s"] = time.perf_counter() - _t

        # ── Step 5: Valve modeling ────────────────────────────────────────
        print("\n[Step 5] Modeling mitral valve...")
        _t = time.perf_counter()
        valve_mesh, valve_info = model_mitral_valve(lv_mesh, output_dir, annulus_method=annulus_method)
        stage["valve_s"] = time.perf_counter() - _t

        # ── Step 6: Cardiac metrics ───────────────────────────────────────
        print("\n[Step 6] Computing cardiac metrics...")
        _t = time.perf_counter()
        metrics = compute_cardiac_metrics(masks_ed["LV"], masks_es["LV"], spacing, output_dir)

        # Add valve info and wall thickness to metrics
        wall_thickness = compute_wall_thickness(lv_mesh, myo_mesh)
        metrics["valve_geometry"]   = valve_info
        metrics["wall_thickness_mm"] = wall_thickness

        # Re-save metrics with added fields
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        stage["metrics_s"] = time.perf_counter() - _t
        stage["core_total_s"] = time.perf_counter() - t_start   # load + meshes + valve + metrics (excludes preview image)
        if timings is not None:
            timings.append(dict(patient=patient_name, **stage))

        # ── Visualization ─────────────────────────────────────────────────
        print("\n[Bonus] Generating mesh preview image...")
        try:
            visualize_meshes(output_dir)
        except Exception as e:
            print(f"  (Preview skipped: {e})")

        # ── Summary ───────────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  ✓ {patient_name} COMPLETE")
        print(f"{'─'*60}")
        print(f"\n  Files saved to: {output_dir}/")
        deliverables = [
            ("full_heart.obj",   "→ Gaussian splatting"),
            ("mitral_valve.obj", "→ Three.js viewer"),
            ("region_map.json",  "→ structure labels"),
            ("metrics.json",     "→ physiology simulation"),
            ("LV.stl",           "→ individual mesh"),
            ("RV.stl",           "→ individual mesh"),
            ("MYO.stl",          "→ individual mesh"),
        ]
        for fname, note in deliverables:
            path = os.path.join(output_dir, fname)
            size = os.path.getsize(path) // 1024 if os.path.exists(path) else 0
            tick = "✓" if os.path.exists(path) else "✗"
            print(f"  {tick} {fname:<22} {note}  [{size} KB]")

        return True

    except Exception as e:
        print(f"\n  ✗ ERROR processing {patient_name}:")
        print(f"    {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Cardiac mesh pipeline")
    parser.add_argument("--patient",  default=None, help="Patient folder name, e.g. patient001")
    parser.add_argument("--all",      action="store_true", help="Process all patients")
    parser.add_argument("--data_dir", default=None, help="Path to ACDC training folder")
    parser.add_argument("--annulus-method", default="lsq", choices=["lsq", "2sigma"],
                        help="annulus ellipse estimator (lsq = least-squares fit, default; 2sigma = first-release heuristic)")
    parser.add_argument("--timing", action="store_true",
                        help="record per-stage wall-clock time per patient and the hardware used (timing.csv, hardware.json)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║  Cardiac Mesh Pipeline                ║")
    print("║  Heart Structure System                  ║")
    print("╚══════════════════════════════════════════╝")

    # Find data directory
    data_dir = args.data_dir or find_data_dir()
    if data_dir is None:
        print("\n ERROR: Cannot find ACDC training data.")
        print("  Expected one of:", DATA_DIRS)
        print("\n  Make sure you downloaded the ACDC dataset and placed it like:")
        print("  data/training/patient001/patient001_frame01.nii.gz")
        sys.exit(1)

    print(f"\n  Data directory : {data_dir}")
    print(f"  Output directory: {OUTPUT_BASE}")

    # Decide which patients to process
    if args.all:
        patient_dirs = sorted(glob.glob(os.path.join(data_dir, "patient*")))
        if not patient_dirs:
            print(f"No patient folders found in {data_dir}")
            sys.exit(1)
        print(f"  Processing {len(patient_dirs)} patients\n")
    else:
        patient_name = args.patient or "patient001"
        patient_path = os.path.join(data_dir, patient_name)
        if not os.path.isdir(patient_path):
            print(f"\n ERROR: Patient folder not found: {patient_path}")
            sys.exit(1)
        patient_dirs = [patient_path]

    # Run pipeline
    start = time.time()
    results = []
    timings = [] if args.timing else None

    for patient_dir in patient_dirs:
        patient_name = os.path.basename(patient_dir)
        out_dir = os.path.join(OUTPUT_BASE, patient_name) if args.all else OUTPUT_BASE
        success = run_patient(patient_dir, out_dir, args.annulus_method, timings)
        results.append((patient_name, success))

    if args.timing and timings:
        os.makedirs(OUTPUT_BASE, exist_ok=True)
        keys = list(timings[0].keys())
        with open(os.path.join(OUTPUT_BASE, "timing.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(timings)
        try:
            cpu = next(l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        except Exception:
            cpu = platform.processor()
        try:
            import psutil; ram = round(psutil.virtual_memory().total / 2**30, 1)
        except Exception:
            ram = None
        import numpy, scipy, skimage, trimesh, sklearn
        hw = dict(cpu=cpu, logical_cores=os.cpu_count(), ram_gb=ram, os=platform.platform(), python=platform.python_version(),
                  numpy=numpy.__version__, scipy=scipy.__version__, scikit_image=skimage.__version__,
                  trimesh=trimesh.__version__, scikit_learn=sklearn.__version__)
        json.dump(hw, open(os.path.join(OUTPUT_BASE, "hardware.json"), "w"), indent=2)
        n = len(timings)
        for k in keys[1:]:
            vals = sorted(t[k] for t in timings)
            print(f"  {k:14s}: mean {sum(vals)/n:7.2f} s  median {vals[n//2]:7.2f} s  min {vals[0]:6.2f}  max {vals[-1]:6.2f}")
        print("  hardware:", hw)

    # Final summary
    elapsed = time.time() - start
    n_ok  = sum(1 for _, ok in results if ok)
    n_err = len(results) - n_ok

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Succeeded: {n_ok} / {len(results)} patients")
    if n_err:
        print(f"  Failed:    {n_err}")
        for name, ok in results:
            if not ok:
                print(f"    ✗ {name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
