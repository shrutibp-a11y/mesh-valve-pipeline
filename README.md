# Cardiac mesh generation and simplified mitral valve surface from ACDC labels

Turns the expert labels of the ACDC training set (short-axis cine CMR, end-diastole and end-systole) into

* surface meshes of the left ventricle (`LV.stl`), myocardium (`MYO.stl`) and right ventricle (`RV.stl`),
* a fused mesh (`full_heart.obj`) with a vertex-range map per structure (`region_map.json`),
* a **simplified parametric mitral valve surface** (`mitral_valve.obj`, 961 vertices, 1,800 triangles),
* cardiac metrics (EDV, ESV, EF, annulus geometry, wall-thickness estimate) in `metrics.json`.

The valve surface is a geometric approximation built from the LV mesh alone (planar annulus ellipse, fixed 4 mm
leaflet bow). It is **not** an anatomical reconstruction, has not been validated against valve imaging, and is not
suitable for finite-element or flow simulation. The pipeline does not segment images: it uses the ACDC ground-truth labels.

## Important correction in this version

See `CHANGELOG.md` for the list of changes and what downstream users must do.


The first release (and the valve results of the first manuscript) assumed that the **base of the left ventricle lies at the
highest Z coordinate**. In the ACDC arrays the base is at the **lowest** Z in 98 of 100 training cases, so the first-release
valve was placed at the apical end. `model_mitral_valve(..., base_end="auto")` now detects the base end from the mesh
geometry (the end with the larger cross-sections). `base_end="high"` together with `annulus_method="2sigma"` reproduces the
first-release outputs exactly. **Valve outputs of the first release should be regenerated** with `python run_all.py --all`.
The valve mesh now also drops the 120 zero-area triangles that the collapsed innermost ring produced (961 vertices, 1,800 triangles).
The basal contour of the LV is about twice as wide as published mitral annulus diameters, so the valve surface is a placeholder
sized to the basal opening of the ventricle, not an estimate of the annulus.

## Annulus estimator

`model_mitral_valve(..., annulus_method=...)` and `run_all.py --annulus-method` accept

| value | meaning |
|---|---|
| `lsq` (default) | direct least-squares ellipse fit (Fitzgibbon et al. 1999; Halir and Flusser 1998) to the XY-projected basal vertices; `a`, `b` are true semi-axes |
| `2sigma` | first-release heuristic `a = 2*sigma1`, `b = 2*sigma2`; reproduces the first-release `metrics.json` values exactly. For ring-like point sets it overestimates the radius by about sqrt(2) (1.40 +/- 0.02 in all 100 ACDC cases) |

## Installation

```bash
python -m venv venv && source venv/bin/activate        # Windows: setup.bat
pip install -r requirements.txt
```

The revision analyses were run with Python 3.12.3, numpy 2.4.4, scipy 1.17.1, scikit-image 0.26.0, scikit-learn 1.8.0,
trimesh 5.1.1, nibabel 5.x, pandas, matplotlib and (only for the self-intersection test) open3d 0.20.0.

## Data

Download the ACDC training set from https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html (registration
required; the data are not redistributed here) and place it as `data/training/patientNNN/`.

## Reproducing the pipeline outputs

```bash
python run_all.py --all --timing          # all 100 patients -> member1/outputs/patientNNN/...
# the LV/MYO/RV meshes are identical in the first and the revised release; only the valve stage changed
```

`--timing` writes `member1/outputs/timing.csv` (seconds per patient for loading, meshing, valve stage, metrics) and
`member1/outputs/hardware.json` (CPU, cores, RAM, OS, library versions). Use `--annulus-method 2sigma` to regenerate the
first-release valve outputs.

## Reproducing the numerical results and figures of the revised paper

```bash
bash revision/reproduce.sh member1/outputs        # step 5 takes about 15-30 s per patient (self-intersection test)
```

| script | produces |
|---|---|
| `00_collect_metrics.py` | per-patient table from all `metrics.json` (group from the official numbering) |
| `01_estimate_annulus.py` | estimators E1 (2 sigma), E2 (least squares), E3 (Feret) for every LV mesh |
| `02_clinical_comparison.py` | normal-group comparison with Ricci et al. reference diameters |
| `03_robustness.py` | translation, rotation, tilt and threshold sensitivity; long-axis tilt |
| `04_associations.py` | pooled, within-group and group-adjusted correlations (Holm-adjusted) |
| `05_mesh_quality.py` | manifoldness, self-intersection, triangle quality of LV, myocardium, RV and first-release valve meshes |
| `05b_volume_and_valve.py` | LV mesh volume (all open ends capped) vs label EDV; quality of the revised valve meshes |
| `08_orientation_check.py` | cross-check of the detected base end with two independent criteria |
| `06_timing_valve_stage.py` | wall-clock time of the valve stage and hardware description |
| `07_make_figures.py` | Fig. 2 and Fig. 3 of the paper |

Intermediate and final results are in `revision/results/`. These scripts regenerate all numerical results and Figs. 2 and 3; the typeset tables, Fig. 1 and the simulation check (a separate module) are not produced here. The group assignment (001-020 DCM, 021-040 HCM, 041-060 MINF,
061-080 NOR, 081-100 ARV) follows the ACDC numbering; check it against the `Info.cfg` files.

## Repository layout

```
run_all.py  find_best_patient.py  requirements.txt  setup.bat  setup.sh
member1/    load_acdc.py  mesh_pipeline.py  valve_modeling.py  compute_metrics.py
revision/   annulus_estimators.py  00_ ... 07_*.py  reproduce.sh  results/
```
