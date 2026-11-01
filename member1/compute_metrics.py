import os
import json
import numpy as np


def compute_cardiac_metrics(lv_mask_ed, lv_mask_es, voxel_spacing, output_dir="member1/outputs"):
    """
    Compute standard echocardiographic measurements.

    What these mean (plain English):
      EDV (End-Diastolic Volume)  = how much blood is in the LV when fully relaxed
      ESV (End-Systolic Volume)   = how much blood remains when fully contracted
      SV  (Stroke Volume)         = how much blood is ejected per beat = EDV - ESV
      EF  (Ejection Fraction)     = percentage of blood ejected = SV/EDV × 100
                                    Normal range: 55–70%
                                    < 40% = heart failure

    Args:
        lv_mask_ed    : 3D binary mask of LV at End-Diastole   (relaxed, max volume)
        lv_mask_es    : 3D binary mask of LV at End-Systole    (contracted, min volume)
        voxel_spacing : (x_mm, y_mm, z_mm) — physical size of each voxel
        output_dir    : where to save metrics.json

    Returns:
        dict with EDV, ESV, SV, EF, and other measurements
    """
    os.makedirs(output_dir, exist_ok=True)

    # Each voxel's volume in mL (1 mL = 1000 mm³)
    voxel_vol_mm3 = float(np.prod(voxel_spacing))
    voxel_vol_mL  = voxel_vol_mm3 / 1000.0

    # Volume = count of voxels × volume per voxel
    EDV_mL = float(lv_mask_ed.sum()) * voxel_vol_mL
    ESV_mL = float(lv_mask_es.sum()) * voxel_vol_mL
    SV_mL  = EDV_mL - ESV_mL

    if EDV_mL == 0:
        raise ValueError("EDV is 0 mL — LV mask at ED is empty. Check your data.")

    EF_pct = (SV_mL / EDV_mL) * 100.0

    # Wall thickness estimation
    # (simple approach: MYO voxel count → average thickness)
    # More accurate methods exist but this is sufficient for our purposes

    metrics = {
        "patient_info": {
            "voxel_spacing_mm": list(voxel_spacing),
            "voxel_volume_mm3": round(voxel_vol_mm3, 4),
        },
        "cardiac_function": {
            "EDV_mL":  round(EDV_mL, 1),
            "ESV_mL":  round(ESV_mL, 1),
            "SV_mL":   round(SV_mL, 1),
            "EF_pct":  round(EF_pct, 1),
            "EF_status": classify_ef(EF_pct),
        },
        "lv_geometry": {
            "ED_voxel_count": int(lv_mask_ed.sum()),
            "ES_voxel_count": int(lv_mask_es.sum()),
        },
        "normal_ranges": {
            "EF_normal_min": 55,
            "EF_normal_max": 70,
            "EDV_normal_range_mL": "80–120",
            "ESV_normal_range_mL": "30–50",
        }
    }

    # Save
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"  EDV : {EDV_mL:.1f} mL  (normal: 80–120 mL)")
    print(f"  ESV : {ESV_mL:.1f} mL  (normal: 30–50 mL)")
    print(f"  SV  : {SV_mL:.1f} mL")
    print(f"  EF  : {EF_pct:.1f}%  ← {classify_ef(EF_pct)}")
    print(f"  metrics.json saved")

    return metrics


def classify_ef(ef_pct):
    """Give a plain-English classification of ejection fraction."""
    if ef_pct >= 55:
        return "Normal (55–70%)"
    elif ef_pct >= 40:
        return "Mildly reduced (40–55%)"
    elif ef_pct >= 30:
        return "Moderately reduced (30–40%)"
    else:
        return "Severely reduced (<30%) — heart failure range"


def compute_wall_thickness(lv_mesh, myo_mesh):
    """
    Estimate average myocardial wall thickness from mesh geometry.
    Uses the bounding box difference as a quick approximation.

    A more accurate method (centerline + ray casting) would be used
    in a production system, but this is sufficient for educational twin.

    Returns:
        estimated average wall thickness in mm
    """
    lv_bounds  = lv_mesh.bounds
    myo_bounds = myo_mesh.bounds

    lv_size  = lv_bounds[1] - lv_bounds[0]
    myo_size = myo_bounds[1] - myo_bounds[0]

    # Thickness ≈ half the difference in radii (crude but fast)
    diff = (myo_size - lv_size) / 2.0
    thickness = float(np.mean(np.abs(diff[:2])))  # average of X and Y
    return round(thickness, 1)


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from member1.load_acdc import load_patient, extract_masks

    search_paths = [
        "data/training/patient001",
        "training/patient001",
    ]
    patient_dir = next((p for p in search_paths if os.path.isdir(p)), None)
    if patient_dir is None:
        print("Cannot find patient data.")
        sys.exit(1)

    print("Loading ED frame...")
    _, labels_ed, spacing = load_patient(patient_dir, frame="ED")
    masks_ed = extract_masks(labels_ed)

    print("Loading ES frame...")
    _, labels_es, _ = load_patient(patient_dir, frame="ES")
    masks_es = extract_masks(labels_es)

    print("\nComputing cardiac metrics...")
    metrics = compute_cardiac_metrics(masks_ed["LV"], masks_es["LV"], spacing)

    print("\ncompute_metrics.py test PASSED")
