import os
import numpy as np
import trimesh
from sklearn.decomposition import PCA


def detect_valve_plane(lv_mesh, percentile=95):
    """
    Find the base of the Left Ventricle — where the mitral valve sits.

    The LV is shaped like a bullet pointing downward.
    The valve is at the TOP (base) of the LV.
    We find it by taking the top N% of vertices by Z-coordinate.

    Args:
        lv_mesh    : trimesh.Trimesh of the Left Ventricle
        percentile : how far up the LV to look (95 = top 5% of vertices)

    Returns:
        base_verts : numpy array of vertices at the LV base
        center     : 3D point — the annulus center
        z_threshold: Z value used as the cutoff
    """
    verts = np.array(lv_mesh.vertices)

    z_threshold = np.percentile(verts[:, 2], percentile)
    base_verts  = verts[verts[:, 2] >= z_threshold]

    if len(base_verts) < 10:
        # Fallback: lower the percentile if we got too few points
        print(f"  Warning: only {len(base_verts)} base vertices at {percentile}th percentile.")
        print(f"  Lowering to 90th percentile.")
        z_threshold = np.percentile(verts[:, 2], 90)
        base_verts  = verts[verts[:, 2] >= z_threshold]

    center = base_verts.mean(axis=0)
    return base_verts, center, z_threshold


def detect_base_end(verts, depths=(0.10, 0.20, 0.30), half_width=1.5):
    """Decide from geometry which end of the LV mesh along Z is the base.

    The LV cavity is widest near the base and tapers to the apex, so the end whose thin-slab
    cross-sections (convex hull of the vertices within +/- half_width mm of a plane, at several
    depths from each end) are larger is the base. In the ACDC arrays the base is at LOW Z in 98
    of 100 training cases, so the Z direction must not be assumed.
    Returns "low" or "high".
    """
    from scipy.spatial import ConvexHull
    z = verts[:, 2]; z0, z1 = z.min(), z.max(); R = z1 - z0
    def area(zc):
        pts = verts[np.abs(z - zc) <= half_width][:, :2]
        return ConvexHull(pts).volume if len(pts) >= 8 else 0.0
    lo = np.mean([area(z0 + f * R) for f in depths]); hi = np.mean([area(z1 - f * R) for f in depths])
    return "low" if lo > hi else "high"


def detect_valve_plane_ex(lv_mesh, percentile=95, base_end="auto"):
    """Like detect_valve_plane but also returns which end was used.

    base_end: "auto" (geometry), "high" (first-release assumption: base at highest Z) or "low".
    Returns (base_verts, center, z_threshold, base_end).
    """
    verts = np.array(lv_mesh.vertices)
    if base_end == "auto":
        base_end = detect_base_end(verts)
    z = verts[:, 2]

    def select(p):
        if base_end == "high":
            thr = np.percentile(z, p); return verts[z >= thr], thr
        thr = np.percentile(z, 100 - p); return verts[z <= thr], thr

    base_verts, z_threshold = select(percentile)
    if len(base_verts) < 10:
        print(f"  Warning: only {len(base_verts)} base vertices at the {percentile}th percentile; lowering to 90th.")
        base_verts, z_threshold = select(90)
    return base_verts, base_verts.mean(axis=0), z_threshold, base_end


def fit_annulus_ellipse(base_verts):
    """
    Fit an ellipse to the annular ring of base vertices using PCA.

    The annulus is the ring-shaped opening of the valve.
    PCA finds the two principal directions of this ring
    (the long axis and short axis of the ellipse).

    Args:
        base_verts : Nx3 array of vertices at the LV base

    Returns:
        axis1      : first principal direction (3D unit vector)
        axis2      : second principal direction (3D unit vector)
        a          : semi-major axis length (mm)
        b          : semi-minor axis length (mm)
    """
    # PCA on the X-Y plane (we care about the shape of the ring, not its height)
    pca = PCA(n_components=2)
    pca.fit(base_verts[:, :2])

    axis1 = np.array([pca.components_[0][0], pca.components_[0][1], 0.0])
    axis2 = np.array([pca.components_[1][0], pca.components_[1][1], 0.0])

    # Semi-axis lengths from the spread of points along each direction
    coords1 = base_verts[:, :2] @ pca.components_[0]
    coords2 = base_verts[:, :2] @ pca.components_[1]
    a = np.std(coords1) * 2.0   # × 2 to get approximate radius, not std
    b = np.std(coords2) * 2.0

    # Make sure a >= b (convention: a is semi-major)
    if b > a:
        axis1, axis2 = axis2, axis1
        a, b = b, a

    return axis1, axis2, a, b


def _fit_ellipse_direct(xy):
    """Direct least-squares ellipse fit (Fitzgibbon et al., 1999; Halir-Flusser stable form).

    Returns (cx, cy, a, b, theta): centre, semi-major a, semi-minor b and the angle (rad) of the
    semi-major axis from +x.
    """
    mx, my = xy[:, 0].mean(), xy[:, 1].mean()
    x, y = xy[:, 0] - mx, xy[:, 1] - my
    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)
    T = -np.linalg.solve(S3, S2.T)
    M = np.linalg.solve(C1, S1 + S2 @ T)
    w, v = np.linalg.eig(M)
    cond = 4 * v[0] * v[2] - v[1] ** 2
    idx = np.where(cond > 0)[0]
    if len(idx) == 0:
        raise ValueError("ellipse fit failed")
    a1 = np.real(v[:, idx[0]])
    A, B, C = a1
    D, E, F = T @ a1
    Mq = np.array([[A, B / 2], [B / 2, C]])
    c = np.linalg.solve(2 * Mq, -np.array([D, E]))
    k = F + 0.5 * (D * c[0] + E * c[1])
    lam, vec = np.linalg.eigh(Mq)
    axes = np.sqrt(np.abs(-k / lam))
    i_major = int(np.argmax(axes))
    major_dir = vec[:, i_major]
    return (float(c[0] + mx), float(c[1] + my), float(axes[i_major]), float(axes[1 - i_major]),
            float(np.arctan2(major_dir[1], major_dir[0])))


def fit_annulus_ellipse_lsq(base_verts):
    """Fit the annulus ellipse to the basal vertices by direct least squares.

    Unlike fit_annulus_ellipse (a = 2*sigma), a and b are genuine semi-axes (radii) of the
    fitted ellipse. Returns (axis1, axis2, a, b, centre_xy).
    """
    cx, cy, a, b, th = _fit_ellipse_direct(base_verts[:, :2])
    axis1 = np.array([np.cos(th), np.sin(th), 0.0])
    axis2 = np.array([-np.sin(th), np.cos(th), 0.0])
    return axis1, axis2, a, b, np.array([cx, cy])


def generate_annulus_ring(center, axis1, axis2, a, b, n_points=120):
    """
    Generate a closed ring of 3D points forming the valve annulus.

    Args:
        center : 3D center point
        axis1  : first ellipse direction
        axis2  : second ellipse direction
        a, b   : semi-axes in mm
        n_points: how many points on the ring (more = smoother)

    Returns:
        ring : Nx3 array of 3D points forming the annulus
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    ring = (
        center
        + a * np.cos(theta)[:, None] * axis1
        + b * np.sin(theta)[:, None] * axis2
    )
    return ring, theta


def generate_leaflets(ring, theta, center, leaflet_height_mm=4.0, n_radial=8, z_sign=-1.0):
    """
    Generate the leaflet surfaces of the mitral valve.

    The mitral valve has two leaflets (anterior and posterior)
    that meet in the middle when closed.
    We model them as a sinusoidal surface bowing from the annulus ring.

    Args:
        ring              : annulus ring points (Nx3)
        theta             : angle values for each ring point
        center            : annulus center
        leaflet_height_mm : how far the leaflets bow downward (mm)
        n_radial          : number of radial subdivisions (more = smoother)

    Returns:
        vertices : all mesh vertices (ring + leaflet points)
        faces    : triangle face indices
    """
    n_ring = len(ring)

    # The leaflets bow towards the ventricle (apex) in a sinusoidal pattern
    # sin(2*theta) creates two lobes — anterior and posterior leaflets
    leaflet_bow = leaflet_height_mm * np.sin(2 * theta)

    # Build multiple radial layers from annulus to center
    all_verts = [ring.copy()]  # layer 0 = annulus

    for r in range(1, n_radial + 1):
        t = r / n_radial   # interpolation factor 0→1
        # Interpolate from annulus ring toward center
        layer = ring * (1 - t) + center * t
        # Apply bow: decreases toward center (leaflets meet at center)
        bow = leaflet_bow * (1 - t) * np.sin(np.pi * t)
        layer[:, 2] += z_sign * bow      # z_sign = -1 if the base is at high Z, +1 if at low Z
        all_verts.append(layer)

    vertices = np.vstack(all_verts)

    # Build triangular faces between adjacent layers
    faces = []
    for layer_i in range(n_radial):
        offset_a = layer_i * n_ring
        offset_b = (layer_i + 1) * n_ring
        for i in range(n_ring):
            j = (i + 1) % n_ring
            # Two triangles per quad
            faces.append([offset_a + i, offset_a + j, offset_b + i])
            faces.append([offset_a + j, offset_b + j, offset_b + i])

    return vertices, np.array(faces)


def model_mitral_valve(lv_mesh, output_dir="member1/outputs",
                        leaflet_height_mm=4.0, n_ring_points=120, n_radial=8,
                        annulus_method="lsq", base_end="auto"):
    """
    Full mitral valve modeling pipeline.

    Steps:
      1. Detect the valve plane at the LV base
      2. Fit an ellipse to the annulus
      3. Generate the annulus ring
      4. Generate parametric leaflet surfaces
      5. Save as mitral_valve.obj

    Args:
        lv_mesh           : trimesh.Trimesh of the Left Ventricle
        output_dir        : where to save the valve mesh
        leaflet_height_mm : how far leaflets bow (tune this per patient)
        n_ring_points     : smoothness of the annulus ring
        n_radial          : smoothness of leaflet radial layers
        base_end          : "auto" detects the base end from the mesh geometry (default); "high" reproduces
                            the first release, which assumed the base at the highest Z (this selected
                            the apical end in 98 of 100 ACDC cases); "low" forces the low-Z end
        annulus_method    : "lsq"    direct least-squares ellipse fit (default; a, b are radii)
                            "2sigma" original heuristic a = 2*sigma1, b = 2*sigma2, which
                                     reproduces the outputs of the first release (its a, b are
                                     about sqrt(2) times larger than the fitted radii)

    Returns:
        valve_mesh : trimesh.Trimesh
        valve_info : dict with annulus measurements (for metrics.json)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("  Step 1: Detecting LV base / valve plane...")
    base_verts, center, z_thresh, base_end_used = detect_valve_plane_ex(lv_mesh, base_end=base_end)
    print(f"    Base end: {base_end_used}-Z end")
    print(f"    Found {len(base_verts)} base vertices at Z ≥ {z_thresh:.1f} mm")
    print(f"    Annulus center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) mm")

    if annulus_method not in ("lsq", "2sigma"):
        raise ValueError("annulus_method must be 'lsq' or '2sigma'")
    print(f"  Step 2: Fitting annulus ellipse ({annulus_method})...")
    if annulus_method == "2sigma":
        axis1, axis2, a, b = fit_annulus_ellipse(base_verts)
    else:
        axis1, axis2, a, b, c_xy = fit_annulus_ellipse_lsq(base_verts)
        center = np.array([c_xy[0], c_xy[1], center[2]])
    print(f"    Semi-major axis a = {a:.1f} mm")
    print(f"    Semi-minor axis b = {b:.1f} mm")
    print(f"    Eccentricity      = {np.sqrt(1 - (b/a)**2):.3f}")

    print("  Step 3: Generating annulus ring...")
    ring, theta = generate_annulus_ring(center, axis1, axis2, a, b, n_ring_points)

    print("  Step 4: Generating leaflet surfaces...")
    vertices, faces = generate_leaflets(ring, theta, center, leaflet_height_mm, n_radial,
                                        z_sign=(-1.0 if base_end_used == 'high' else 1.0))

    # Build and clean the valve mesh
    valve_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    valve_mesh.merge_vertices()
    # the innermost radial layer collapses onto the centre: drop the resulting zero-area triangles
    # (120 of 1,920 with the default settings; the first release kept them)
    valve_mesh.update_faces(valve_mesh.nondegenerate_faces())
    valve_mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(valve_mesh)

    # Save
    output_path = os.path.join(output_dir, "mitral_valve.obj")
    valve_mesh.export(output_path)
    print(f"  mitral_valve.obj saved — {len(valve_mesh.vertices)} vertices")

    # Info dict for metrics.json
    valve_info = {
        "annulus_center_mm":    center.tolist(),
        "semi_major_axis_mm":   round(a, 2),
        "semi_minor_axis_mm":   round(b, 2),
        "annulus_area_mm2":     round(np.pi * a * b, 1),
        "leaflet_height_mm":    leaflet_height_mm,
        "z_threshold_mm":       round(float(z_thresh), 2),
        "method":               "parametric_pca_sinusoidal",
        "annulus_method":       annulus_method,
        "base_end":             base_end_used,
        "description":          "Simplified parametric valve surface from ventricular geometry",
    }

    return valve_mesh, valve_info


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load an existing LV mesh (run mesh_pipeline.py first)
    lv_path = "member1/outputs/LV.stl"
    if not os.path.exists(lv_path):
        print("LV.stl not found. Run mesh_pipeline.py first.")
        sys.exit(1)

    print(f"Loading LV mesh from {lv_path}")
    lv_mesh = trimesh.load(lv_path)
    print(f"LV mesh: {len(lv_mesh.vertices):,} vertices\n")

    print("Running valve modeling pipeline...")
    valve_mesh, valve_info = model_mitral_valve(lv_mesh)

    print("\nValve measurements:")
    for k, v in valve_info.items():
        print(f"  {k}: {v}")

    print("\nvalve_modeling.py test PASSED")
