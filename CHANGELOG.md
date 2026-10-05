# Changelog

## Revision of October 2026 (read this if you use `mitral_valve.obj` or `metrics.json`)

### What was wrong in the first release
- The valve code assumed that the **base of the left ventricle is at the highest Z** coordinate. In the ACDC arrays the base is at
  the **lowest** Z in 98 of 100 training cases (patients 037 and 047 are the exceptions; patients 001, 037 and 047 are ambiguous
  by independent criteria). The valve was therefore built at the **apical** end.
- The annulus semi-axes were `2 * standard deviation`, which overestimates a radius by about the square root of 2.
- The valve mesh contained 120 zero-area triangles at its centre (1,920 faces, of which 120 degenerate).

### What changed
- `model_mitral_valve(..., base_end="auto")` detects the base end from the mesh geometry. `base_end="high"` with
  `annulus_method="2sigma"` reproduces the first release exactly.
- `annulus_method="lsq"` (default) fits the ellipse by direct least squares, so `semi_major_axis_mm` and `semi_minor_axis_mm`
  in `metrics.json` are now true radii. The old values are about 1.4 times larger.
- The valve mesh has 961 vertices and 1,800 faces (degenerate triangles removed).
- `metrics.json -> valve_geometry` now also records `base_end` and `annulus_method`.
- An internal label field in `metrics.json -> valve_geometry` was replaced by a plain `description` field, and console text was simplified. No numerical value changed.
- `run_all.py` has `--timing` (per-stage times and hardware description) and `--annulus-method`.

### What did NOT change
- `LV.stl`, `RV.stl`, `MYO.stl`, `full_heart.obj`, `region_map.json`.
- EDV, ESV and EF (they come from label voxel counts, not from the valve stage).

### What you must do
- **Do not use the valve outputs of the first release.** Regenerate with `python run_all.py --all`.
- Even after correction, the valve is sized to the **basal opening of the left ventricle**, which is about twice the width of a
  real mitral annulus (normal group: 51.7 mm equivalent diameter against 26 to 29 mm in the literature). It is a placeholder for
  visualisation and simulation initialisation, not an anatomical model and not an annulus measurement.
