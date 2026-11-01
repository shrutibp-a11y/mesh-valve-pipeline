import os
import json
import numpy as np
import trimesh
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter, zoom
import matplotlib
matplotlib.use('Agg')  # no popup windows ever
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
 
 
def mask_to_mesh(mask, voxel_spacing, label_name, output_dir="member1/outputs"):
    os.makedirs(output_dir, exist_ok=True)

    if mask.sum() == 0:
        raise ValueError(f"Mask for {label_name} is empty.")

    from scipy.ndimage import gaussian_filter, zoom, binary_fill_holes, binary_closing
    from skimage.morphology import ball

    # Step 1: Fill holes in the mask first
    mask = binary_fill_holes(mask).astype(np.uint8)

    # Step 2: Upsample Z axis aggressively
    zoom_factors = (1.0, 1.0, voxel_spacing[2] / voxel_spacing[0])
    mask_up = zoom(mask.astype(np.float32), zoom_factors, order=1)

    # Step 3: Binary closing to fill gaps between slices
    mask_closed = binary_closing(mask_up > 0.5, ball(3)).astype(np.float32)

    # Step 4: Strong Gaussian smoothing
    mask_smooth = gaussian_filter(mask_closed, sigma=(2.0, 2.0, 3.0))

    # Step 5: Re-threshold
    mask_smooth = (mask_smooth > 0.2).astype(np.uint8)

    # Step 6: Isotropic spacing for marching cubes
    iso = (voxel_spacing[0], voxel_spacing[0], voxel_spacing[0])
    verts, faces, normals, _ = marching_cubes(
        mask_smooth, level=0.5, spacing=iso, allow_degenerate=False)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh = trimesh.util.concatenate([mesh])
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)

    output_path = os.path.join(output_dir, f"{label_name}.stl")
    mesh.export(output_path)
    print(f"  {label_name}.stl saved — {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    return mesh
 
def fuse_meshes(lv_mesh, rv_mesh, myo_mesh, output_dir="member1/outputs"):
    os.makedirs(output_dir, exist_ok=True)
 
    lv_count  = len(lv_mesh.vertices)
    myo_count = len(myo_mesh.vertices)
 
    combined = trimesh.util.concatenate([lv_mesh, myo_mesh, rv_mesh])
    combined.export(os.path.join(output_dir, "full_heart.obj"))
    print(f"  full_heart.obj saved — {len(combined.vertices):,} total vertices")
 
    region_map = {
        "LV":  {"label_id": 3, "color_hex": "#E84040", "description": "Left Ventricle",
                "vertex_start": 0, "vertex_end": lv_count},
        "MYO": {"label_id": 2, "color_hex": "#FF8C00", "description": "Myocardium",
                "vertex_start": lv_count, "vertex_end": lv_count + myo_count},
        "RV":  {"label_id": 1, "color_hex": "#4488FF", "description": "Right Ventricle",
                "vertex_start": lv_count + myo_count, "vertex_end": len(combined.vertices)},
    }
    with open(os.path.join(output_dir, "region_map.json"), "w") as f:
        json.dump(region_map, f, indent=2)
    print(f"  region_map.json saved")
    return combined
 
 
def visualize_meshes(output_dir="member1/outputs"):
    fig = plt.figure(figsize=(15, 5))
    structures = [("LV.stl", "#E84040", "Left Ventricle"),
                  ("MYO.stl", "#FF8C00", "Myocardium"),
                  ("RV.stl",  "#4488FF", "Right Ventricle")]
 
    for i, (filename, color, title) in enumerate(structures):
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            continue
        mesh = trimesh.load(path)
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        sample = mesh.faces[::max(1, len(mesh.faces) // 500)]
        triangles = mesh.vertices[sample]
        poly = Poly3DCollection(triangles, alpha=0.4, facecolor=color, edgecolor="none")
        ax.add_collection3d(poly)
        bounds = mesh.bounds
        for dim, (lo, hi) in enumerate(zip(bounds[0], bounds[1])):
            pad = (hi - lo) * 0.1
            getattr(ax, f"set_{'xyz'[dim]}lim")(lo - pad, hi + pad)
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
 
    plt.suptitle("Heart Structure Meshes", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "mesh_preview.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Mesh preview saved: {save_path}")