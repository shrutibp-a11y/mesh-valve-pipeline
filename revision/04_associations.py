"""Pooled, within-group and group-adjusted associations of annulus area (E2) with EDV and EF.
usage: python revision/04_associations.py [results_dir]"""
import sys, os, numpy as np, pandas as pd
from scipy import stats
RES = sys.argv[1] if len(sys.argv) > 1 else "revision/results"
d = pd.read_csv(os.path.join(RES, "estimates.csv")); groups = ["NOR", "DCM", "HCM", "MINF", "ARV"]
def partial(x, y, g):
    D = pd.get_dummies(g).values.astype(float)
    rx = x - D @ np.linalg.lstsq(D, x, rcond=None)[0]; ry = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]; df = len(x) - 2 - (D.shape[1] - 1)
    p = 2 * stats.t.sf(abs(r * np.sqrt(df / (1 - r ** 2))), df); return r, p, df
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, idx in enumerate(o):
        run = max(run, (m - i) * ps[idx]); adj[idx] = min(1, run)
    return adj
rows = []
for y in ("EDV", "EF"):
    r, p = stats.pearsonr(d.area2, d[y]); rows.append(dict(y=y, subset="pooled", n=len(d), r=r, p=p))
    rs, ps = stats.spearmanr(d.area2, d[y]); rows.append(dict(y=y, subset="pooled_spearman", n=len(d), r=rs, p=ps))
    pw = []
    for g in groups:
        s = d[d.group == g]; r, p = stats.pearsonr(s.area2, s[y]); rows.append(dict(y=y, subset=g, n=len(s), r=r, p=p)); pw.append(p)
    for g, a in zip(groups, holm(np.array(pw))):
        for row in rows:
            if row["y"] == y and row["subset"] == g: row["p_holm"] = a
    r, p, dfp = partial(d.area2.values, d[y].values, d.group.values)
    rows.append(dict(y=y, subset="group_adjusted", n=len(d), r=r, p=p, df=dfp))
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "associations.csv"), index=False); print(R.round(3).to_string())
fe, pe = stats.f_oneway(*[d[d.group == g].EF for g in groups]); print("ANOVA EF: F=%.1f p=%.1e" % (fe, pe))
