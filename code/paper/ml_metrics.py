#!/usr/bin/env python3
"""
ml_metrics.py -- ML d-DOS accuracy for the paper, from the HPC export package (no HPC access needed).

SOURCE  $HEUSLER_TE_EXPORT (built on Aurora 2026-07-19 by make_dos_export.py):
        metadata/metrics/<family>/fold<k>/per_sample_metrics.csv  -- per-material test metrics, 5-fold CV
        runtime_assets/heldout_dumps/<family>/fold<k>/test_{predictions,targets}.npy, test_ids.txt
CAVEATS (state in the paper): metrics are from each fold's best-validation checkpoint (best.pt), not the deployed
SWA ensemble; whether a multi-family pre-trained initialisation saw the test folds cannot be verified without HPC logs.

OUTPUT  results/paper_analysis/ml_metrics.xlsx, ml_metrics.txt
  per family: n, per-sample Pearson r (mean, sd, median, p5), Fermi-window r, RMSE, MAE (states/eV/atom),
  d-DOS gap classification (true/pred gapped: accuracy, majority baseline, balanced accuracy, MCC, recall, precision),
  DOS at E_F (sum of both spins at index 1000/3001): Pearson r and MAE from the held-out dumps.
"""
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
import glob
import numpy as np
import pandas as pd

X = _HTE                                   # export package root
OUT = _HTR + "/results/paper_analysis"
FAMS = ["full_cubic", "full_tetra", "half_cubic", "half_tetra", "inverse_cubic", "inverse_tetra"]
I_EF = (1000, 3001)                                                         # E_F index in the spin-up / spin-down halves


def cls_scores(t, p):
    """Binary classification scores for the d-DOS 'gapped' label (t, p boolean arrays)."""
    t, p = np.asarray(t, bool), np.asarray(p, bool)
    tp, tn = (t & p).sum(), (~t & ~p).sum(); fp, fn = (~t & p).sum(), (t & ~p).sum()
    rec = tp / (tp + fn) if tp + fn else np.nan                              # recall (sensitivity) on gapped
    spec = tn / (tn + fp) if tn + fp else np.nan                             # specificity on metallic
    prec = tp / (tp + fp) if tp + fp else np.nan
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return dict(n_true_gapped=int(t.sum()), accuracy=(tp + tn) / len(t), majority_baseline=max(t.mean(), 1 - t.mean()),
                balanced_accuracy=np.nanmean([rec, spec]), MCC=(tp * tn - fp * fn) / den if den else np.nan,
                recall_gapped=rec, precision_gapped=prec)


def main():
    rows = []
    for fam in FAMS:
        ps = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{X}/metadata/metrics/{fam}/fold*/per_sample_metrics.csv"))])
        r = dict(family=fam, n_test=len(ps), unique_ids=ps.material_id.nunique(),
                 pearson_mean=ps.pearson_r.mean(), pearson_sd=ps.pearson_r.std(), pearson_median=ps.pearson_r.median(),
                 pearson_p5=ps.pearson_r.quantile(0.05), fermi2eV_pearson=ps.fermi_pearson_2eV.mean(),
                 rmse=ps.rmse.mean(), mae=ps.mae.mean())
        r.update(cls_scores(ps.true_gapped, ps.pred_gapped))
        gp, gt = [], []
        for fd in sorted(glob.glob(f"{X}/runtime_assets/heldout_dumps/{fam}/fold*")):   # DOS at E_F from raw dumps
            pr = np.load(f"{fd}/test_predictions.npy", mmap_mode="r"); tg = np.load(f"{fd}/test_targets.npy", mmap_mode="r")
            gp.append(pr[:, I_EF[0]] + pr[:, I_EF[1]]); gt.append(tg[:, I_EF[0]] + tg[:, I_EF[1]])
        gp, gt = np.concatenate(gp), np.concatenate(gt)
        r.update(gEF_pearson=np.corrcoef(gp, gt)[0, 1], gEF_mae=np.abs(gp - gt).mean(), gEF_n=len(gp))
        rows.append(r)
    t = pd.DataFrame(rows)
    allps = pd.concat([pd.read_csv(f) for f in glob.glob(f"{X}/metadata/metrics/*/fold*/per_sample_metrics.csv")])
    tot = dict(family="ALL", n_test=len(allps), unique_ids=allps.material_id.nunique(), pearson_mean=allps.pearson_r.mean(),
               pearson_sd=allps.pearson_r.std(), pearson_median=allps.pearson_r.median(), pearson_p5=allps.pearson_r.quantile(0.05),
               fermi2eV_pearson=allps.fermi_pearson_2eV.mean(), rmse=allps.rmse.mean(), mae=allps.mae.mean())
    tot.update(cls_scores(allps.true_gapped, allps.pred_gapped))
    t = pd.concat([t, pd.DataFrame([tot])], ignore_index=True)
    t.to_excel(f"{OUT}/ml_metrics.xlsx", index=False)
    txt = t.round(4).to_string(index=False)
    open(f"{OUT}/ml_metrics.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
