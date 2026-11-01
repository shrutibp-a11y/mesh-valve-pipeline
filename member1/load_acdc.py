import os
import glob
import nibabel as nib
import numpy as np


# ─── ACDC label convention ────────────────────────────────────────────────────
# 0 = background
# 1 = Right Ventricle  (RV)
# 2 = Myocardium       (MYO)  — the heart wall muscle
# 3 = Left Ventricle   (LV)
# ──────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "LV":  3,
    "MYO": 2,
    "RV":  1,
}


def find_frames(patient_dir):
    """
    Find all frame files in a patient directory.
    Returns a dict: { 'ED': (img_path, gt_path), 'ES': (img_path, gt_path) }
    ACDC stores end-diastole (ED) and end-systole (ES) frames as separate .nii.gz files.
    """
    gt_files = sorted(glob.glob(os.path.join(patient_dir, "*_gt.nii")) + 
                  glob.glob(os.path.join(patient_dir, "*_gt.nii.gz")))
    if len(gt_files) < 2:
        raise FileNotFoundError(
            f"Expected at least 2 GT frames in {patient_dir}, found {len(gt_files)}.\n"
            "Make sure you downloaded the ACDC training set correctly."
        )

    frames = {}
    for i, gt_path in enumerate(gt_files):
        if gt_path.endswith("_gt.nii.gz"):
            img_path = gt_path.replace("_gt.nii.gz", ".nii.gz")
        else:
            img_path = gt_path.replace("_gt.nii", ".nii")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        key = "ED" if i == 0 else "ES"
        frames[key] = (img_path, gt_path)

    return frames


def load_patient(patient_dir, frame="ED"):
    """
    Load one frame from a patient directory.

    Args:
        patient_dir: path to patient folder, e.g. "data/training/patient001"
        frame:       "ED" (end-diastole, heart relaxed) or "ES" (end-systole, heart contracted)

    Returns:
        mri_data   : 3D numpy array of MRI intensities
        label_data : 3D numpy array of integer labels (0,1,2,3)
        voxel_size : tuple of (x_mm, y_mm, z_mm) — physical size of each voxel
    """
    frames = find_frames(patient_dir)
    img_path, gt_path = frames[frame]

    img = nib.load(img_path)
    gt  = nib.load(gt_path)

    mri_data   = img.get_fdata().astype(np.float32)
    label_data = gt.get_fdata().astype(np.uint8)
    voxel_size = tuple(float(v) for v in img.header.get_zooms()[:3])

    # Sanity checks
    assert mri_data.shape == label_data.shape, (
        f"MRI shape {mri_data.shape} != label shape {label_data.shape}"
    )
    assert any(label_data.max() > 0 for _ in [1]), (
        "All labels are 0 — check your data files"
    )

    return mri_data, label_data, voxel_size


def extract_masks(label_data):
    """
    Convert the integer label array into separate binary masks.

    Args:
        label_data: 3D uint8 array with values 0,1,2,3

    Returns:
        dict with keys 'LV', 'MYO', 'RV', each a binary uint8 array
    """
    masks = {}
    for name, label_id in LABEL_MAP.items():
        mask = (label_data == label_id).astype(np.uint8)
        if mask.sum() == 0:
            print(f"  WARNING: {name} mask is empty (all zeros). Check label values.")
        masks[name] = mask
    return masks


def normalize_intensity(mri_data):
    """
    Normalize MRI intensities to [0, 1] range.
    Simple min-max normalization — helps with visualization.
    """
    min_val = mri_data.min()
    max_val = mri_data.max()
    if max_val == min_val:
        return np.zeros_like(mri_data)
    return (mri_data - min_val) / (max_val - min_val)


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Try to find a patient folder
    search_paths = [
        "data/training/patient001",
        "training/patient001",
        "../data/training/patient001",
    ]

    patient_dir = None
    for p in search_paths:
        if os.path.isdir(p):
            patient_dir = p
            break

    if patient_dir is None:
        print("Could not find patient001 folder.")
        print("Run this script from your project root, or edit the search_paths list above.")
        print("Expected structure: data/training/patient001/patient001_frame01.nii.gz")
        sys.exit(1)

    print(f"Loading from: {patient_dir}")
    print()

    for frame in ["ED", "ES"]:
        mri, labels, spacing = load_patient(patient_dir, frame=frame)
        masks = extract_masks(labels)
        print(f"Frame {frame}:")
        print(f"  MRI shape     : {mri.shape}")
        print(f"  Voxel spacing : {spacing} mm")
        print(f"  LV  voxels    : {masks['LV'].sum()}")
        print(f"  MYO voxels    : {masks['MYO'].sum()}")
        print(f"  RV  voxels    : {masks['RV'].sum()}")
        print()

    print("load_acdc.py test PASSED")
