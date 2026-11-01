"""
Annulus (basal-opening) estimators used in the revision analyses.

E1  original heuristic of the submitted paper: a = 2*sigma1, b = 2*sigma2 of the
    XY-projected basal vertices along the two PCA directions.
E2  direct least-squares ellipse fit (Fitzgibbon et al. 1999; numerically stable
    form of Halir and Flusser 1998) to the same XY-projected basal vertices.
E3  Feret diameters (maximum / minimum caliper width) of the convex hull of the
    same points.

All functions take an (N, 3) vertex array of an LV surface mesh in millimetres and
return plain Python/NumPy values, so they can be re-used outside the pipeline.
"""
import numpy as np
from scipy.spatial import ConvexHull


def basal_vertices(verts, percentile=95):
    """Vertices at or above the given percentile of Z (same rule as valve_modeling.py)."""
    z_thr = np.percentile(verts[:, 2], percentile)
    base = verts[verts[:, 2] >= z_thr]
    if len(base) < 10:
        z_thr = np.percentile(verts[:, 2], 90)
        base = verts[verts[:, 2] >= z_thr]
    return base, float(z_thr)


def two_sigma_axes(base):
    """E1: the heuristic used in the submitted paper (a = 2*sigma1, b = 2*sigma2)."""
    xy = base[:, :2]
    xyc = xy - xy.mean(axis=0)
    cov = np.cov(xyc.T)
    w, v = np.linalg.eigh(cov)          # ascending
    order = np.argsort(w)[::-1]
    sig = np.sqrt(w[order])
    return 2.0 * sig[0], 2.0 * sig[1]


def fit_ellipse_lsq(xy):
    """E2: direct least-squares ellipse fit (Halir-Flusser). Returns (a, b), a >= b."""
    x = xy[:, 0] - xy[:, 0].mean()
    y = xy[:, 1] - xy[:, 1].mean()
    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    M = np.linalg.solve(C1, M)
    w, v = np.linalg.eig(M)
    cond = 4 * v[0] * v[2] - v[1] ** 2
    idx = np.where(cond > 0)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    a1 = np.real(v[:, idx[0]])
    a2 = T @ a1
    A, B, C = a1
    Dd, E, F = a2
    M0 = np.array([[F, Dd / 2, E / 2], [Dd / 2, A, B / 2], [E / 2, B / 2, C]])
    Mq = np.array([[A, B / 2], [B / 2, C]])
    lam = np.sort(np.linalg.eigvalsh(Mq))
    d0, dq = np.linalg.det(M0), np.linalg.det(Mq)
    axes = np.sqrt(np.abs(-d0 / (dq * lam)))
    return float(max(axes)), float(min(axes))


def feret_diameters(xy):
    """E3: maximum and minimum Feret diameters of the convex hull of the points."""
    hull = ConvexHull(xy)
    pts = xy[hull.vertices]
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    fmax = d.max()
    widths = []
    n = len(pts)
    for i in range(n):
        e = pts[(i + 1) % n] - pts[i]
        nrm = np.array([-e[1], e[0]])
        nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
        proj = pts @ nrm
        widths.append(proj.max() - proj.min())
    return float(fmax), float(min(widths))


def rotation_matrix(axis, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def base_end(verts, depths=(0.10, 0.20, 0.30), half_width=1.5):
    """Decide which end of the LV mesh along Z is the base, from geometry alone.

    The LV cavity is widest near the base and tapers towards the apex, so the end whose thin-slab
    cross-sections (convex hull of the vertices within +/- half_width mm of the plane) are larger
    is the base. Cross-sections are taken at several depths from each end and averaged.
    Returns ('low' | 'high', mean_area_low_end, mean_area_high_end) with areas in mm^2.
    """
    z = verts[:, 2]; z0, z1 = z.min(), z.max(); R = z1 - z0
    def area(zc):
        s = verts[np.abs(z - zc) <= half_width][:, :2]
        return ConvexHull(s).volume if len(s) >= 8 else 0.0
    lo = float(np.mean([area(z0 + f * R) for f in depths])); hi = float(np.mean([area(z1 - f * R) for f in depths]))
    return ("low" if lo > hi else "high"), lo, hi


def orient_base_high(verts):
    """Mirror Z (z -> zmin + zmax - z) if the base is at the low-Z end, so that the base is always at high Z."""
    end, lo, hi = base_end(verts)
    if end == "low":
        v = np.array(verts, copy=True); v[:, 2] = verts[:, 2].min() + verts[:, 2].max() - verts[:, 2]
        return v, "low"
    return verts, "high"


def estimate_all(verts, percentile=95, orient=True):
    """Run E1, E2 and E3 on one LV mesh. Areas use pi*a*b (semi-axes), mm and mm^2."""
    end = "n/a"
    if orient:
        verts, end = orient_base_high(verts)
    base, z_thr = basal_vertices(verts, percentile)
    a1, b1 = two_sigma_axes(base)
    a2, b2 = fit_ellipse_lsq(base[:, :2])
    fmax, fmin = feret_diameters(base[:, :2])
    return dict(base_end=end, n_base=len(base), z_thr=z_thr,
                slab_mm=float(base[:, 2].max() - base[:, 2].min()),
                a1=a1, b1=b1, area1=np.pi * a1 * b1,
                a2=a2, b2=b2, area2=np.pi * a2 * b2,
                feret_max=fmax, feret_min=fmin)


if __name__ == "__main__":
    # Unit test of the ellipse fit on synthetic data.
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 800)
    A, B, rot = 30.0, 20.0, np.radians(37)
    x = A * np.cos(th)
    y = B * np.sin(th)
    X = x * np.cos(rot) - y * np.sin(rot) + 5
    Y = x * np.sin(rot) + y * np.cos(rot) - 3
    pts = np.column_stack([X, Y]) + rng.normal(0, 0.3, (800, 2))
    print("LSQ ellipse (expect ~30, ~20):", fit_ellipse_lsq(pts))
    # ring-like points: 2*sigma should over-estimate the radius by about sqrt(2)
    r = 25.0
    ang = rng.uniform(0, 2 * np.pi, 5000)
    ring = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
    base = np.column_stack([ring, np.zeros(len(ring))])
    a1, b1 = two_sigma_axes(base)
    print("ring r=25: 2sigma axes", round(a1, 2), round(b1, 2), "ratio to r", round(a1 / r, 3),
          "(sqrt2 = 1.414)")
    print("ring LSQ:", fit_ellipse_lsq(ring))



def boundary_loops(mesh):
    """All closed boundary loops of a mesh as lists of vertex indices."""
    u, c = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    b = u[c == 1]
    adj = {}
    for a, bb in b:
        adj.setdefault(int(a), []).append(int(bb)); adj.setdefault(int(bb), []).append(int(a))
    seen, loops = set(), []
    for s in list(adj):
        if all((min(s, n), max(s, n)) in seen for n in adj[s]):
            continue
        lp, cur = [s], s
        while True:
            nx = [n for n in adj[cur] if (min(cur, n), max(cur, n)) not in seen]
            if not nx:
                break
            n = nx[0]; seen.add((min(cur, n), max(cur, n)))
            if n == s:
                break
            lp.append(n); cur = n
        if len(lp) > 2:
            loops.append(lp)
    return loops


def boundary_loop(mesh, return_indices=False):
    """Longest closed boundary loop of an open mesh, as an (N, 3) vertex array (or None).
    With return_indices=True the ordered vertex indices are returned instead."""
    e = mesh.edges_sorted
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    bedges = uniq[counts == 1]
    if len(bedges) < 6:
        return None
    adj = {}
    for a, b in bedges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    seen_edges = set()
    loops = []
    for start in list(adj.keys()):
        if all((min(start, n), max(start, n)) in seen_edges for n in adj[start]):
            continue
        loop = [start]
        prev, cur = None, start
        while True:
            nxts = [n for n in adj[cur] if (min(cur, n), max(cur, n)) not in seen_edges]
            if not nxts:
                break
            nxt = nxts[0]
            seen_edges.add((min(cur, nxt), max(cur, nxt)))
            if nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) > 5:
            loops.append(loop)
    if not loops:
        return None
    verts = np.asarray(mesh.vertices)
    def perim(l):
        p = verts[l]
        return np.sum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1))
    best = max(loops, key=perim)
    return np.asarray(best) if return_indices else verts[best]


def boundary_loop_ellipse(mesh):
    """E4: orientation-free estimator based on the mesh's own basal boundary loop.

    1. take the longest boundary loop of the open LV surface,
    2. fit a plane to it (SVD) and project the loop into that plane,
    3. fit a direct least-squares ellipse in the plane.
    Nothing here refers to the image axes or to a Z-percentile, so the estimate is invariant to
    rotation and translation of the mesh. Returns a dict with semi-axes, plane tilt w.r.t. Z,
    planarity (RMS distance to the plane) and the number of loop vertices.
    """
    loop = boundary_loop(mesh)
    if loop is None:
        return None
    c = loop.mean(axis=0)
    u, s, vt = np.linalg.svd(loop - c)
    normal = vt[2]
    basis = vt[:2]
    xy = (loop - c) @ basis.T
    a, b = fit_ellipse_lsq(xy)
    rms = float(np.sqrt(np.mean(((loop - c) @ normal) ** 2)))
    tilt = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1))))
    fmax, fmin = feret_diameters(xy)
    return dict(a4=a, b4=b, area4=np.pi * a * b, plane_tilt=tilt, planarity_rms=rms,
                n_loop=len(loop), loop_feret_max=fmax, loop_feret_min=fmin)


def fit_ellipse_lsq_full(xy):
    """Direct least-squares ellipse fit returning (cx, cy, a, b, theta).

    theta is the angle (rad) of the semi-major axis a from the +x axis; (cx, cy) is the ellipse
    centre in the same coordinates as `xy`. Same algorithm as `fit_ellipse_lsq`.
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
    # centre: solve gradient = 0
    c = np.linalg.solve(2 * Mq, -np.array([D, E]))
    # value of the quadratic form at the centre
    k = F + 0.5 * (D * c[0] + E * c[1])
    lam, vec = np.linalg.eigh(Mq)             # ascending eigenvalues
    axes = np.sqrt(np.abs(-k / lam))          # semi-axis along eigenvector i
    i_major = int(np.argmax(axes))
    a, b = axes[i_major], axes[1 - i_major]
    major_dir = vec[:, i_major]
    theta = float(np.arctan2(major_dir[1], major_dir[0]))
    return float(c[0] + mx), float(c[1] + my), float(a), float(b), theta
