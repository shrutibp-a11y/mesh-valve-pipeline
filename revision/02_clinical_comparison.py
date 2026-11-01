"""Compare annulus estimators (normal group) with Ricci et al. (JCMR 2021; 23:5) reference diameters:
end-diastolic antero-posterior (3-chamber) diameter, men 2.9 +/- 0.4 cm, women 2.6 +/- 0.4 cm.
usage: python revision/02_clinical_comparison.py [results_dir]"""
import sys, os, numpy as np, pandas as pd
from scipy import stats
RES = sys.argv[1] if len(sys.argv) > 1 else "revision/results"
d = pd.read_csv(os.path.join(RES, "estimates.csv")); nor = d[d.group == "NOR"]
cand = {
 "First release: 2-sigma semi-axis a (highest-Z slab)": nor.a1_top,
 "First release: equivalent diameter, least squares (highest-Z slab)": 2 * np.sqrt(nor.a2_top * nor.b2_top),
 "Base end: 2-sigma semi-axis a (E1)": nor.a1,
 "Base end: major diameter 2a (E2)": 2 * nor.a2, "Base end: minor diameter 2b (E2)": 2 * nor.b2,
 "Base end: equivalent diameter 2*sqrt(a*b) (E2)": 2 * np.sqrt(nor.a2 * nor.b2),
 "Base end: Feret maximum (E3)": nor.feret_max, "Base end: Feret minimum (E3)": nor.feret_min}
rows = []
for k, v in cand.items():
    r = {"estimator": k, "mean": v.mean(), "sd": v.std(ddof=1)}
    for name, ref in (("men_29", 29.0), ("women_26", 26.0)):
        t, p = stats.ttest_1samp(v, ref); ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean() - ref, scale=stats.sem(v))
        r.update({f"diff_{name}": v.mean() - ref, f"ci_lo_{name}": ci[0], f"ci_hi_{name}": ci[1], f"t_{name}": t, f"p_{name}": p})
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "annulus_vs_reference.csv"), index=False)
print(R[["estimator", "mean", "sd", "diff_men_29", "diff_women_26", "p_men_29", "p_women_26"]].round(2).to_string())
eq = 2 * np.sqrt(nor.a2 * nor.b2); print("ratio base-end equivalent diameter / reference: men %.2f, women %.2f" % (eq.mean() / 29, eq.mean() / 26))
