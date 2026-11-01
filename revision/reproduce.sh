#!/usr/bin/env bash
# Regenerates every table, figure and number of the revised paper from the pipeline outputs.
# usage (repository root):  bash revision/reproduce.sh [outputs_dir]   e.g. member1/outputs
# Add --skip-selfintersection to step 5 for a fast run (the Open3D self-intersection test takes
# roughly 15-30 s per patient on one CPU core).
set -euo pipefail
OUT="${1:-member1/outputs}"; RES="revision/results"; mkdir -p "$RES"
python revision/00_collect_metrics.py   "$OUT" "$RES"
python revision/01_estimate_annulus.py  "$OUT" "$RES"
python revision/02_clinical_comparison.py "$RES"
python revision/03_robustness.py        "$OUT" "$RES"
python revision/04_associations.py      "$RES"
python revision/05_mesh_quality.py      "$OUT" "$RES"      # LV/MYO/RV/first-release valve; slow (self-intersection test)
python revision/05b_volume_and_valve.py "$OUT" "$RES"      # LV volume vs EDV, revised valve mesh quality
python revision/08_orientation_check.py "$OUT" "$RES"
python revision/06_timing_valve_stage.py "$OUT" "$RES"
python revision/07_make_figures.py      "$OUT" "$RES"
echo "Done. Results in $RES"
