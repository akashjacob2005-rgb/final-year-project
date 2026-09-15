"""External validation of the deployed MRI model on OASIS-1 (never-seen data).

OASIS-1 is a DIFFERENT study from the OASIS-3 training data: 1.5T scanners
(vs 3T), 2007-era acquisition, ANALYZE format. Domain shift is expected — the
point of this script is to MEASURE it honestly.

Volume choice: PROCESSED/MPRAGE/SUBJ_111 (N4-corrected, 1mm, native anatomy,
not atlas-warped, not skull-stripped) — the closest analogue to a raw upload.

Orientation: OASIS-1 ANALYZE headers misreport orientation (the "canonical"
volume's axis 2 is actually sagittal). Verified visually: the true axial
plane is axis 1, and `data[:, idx, :]` is already anterior-up / radiological
left-right — matching the training slice orientation with NO extra transform.

Inference matches the deployed API exactly: same slice fractions and
percentile normalisation as ml/mri/preprocess.py, same resize/normalise
batch prep, same softmax-averaged horizontal-flip TTA, same slice-mean
aggregation (reuses backend/app/ml/mri_model.py helpers).

Run (from repo root):
    backend/.venv/bin/python ml/mri/external_validate.py \
        --oasis1-dir ~/Downloads/7th_sem/PP-2/external/oasis1/disc1
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app.ml import mri_model  # noqa: E402  (pulls ml/ onto sys.path too)
from mri import preprocess  # noqa: E402


def read_cdr(txt_path: Path) -> float | None:
    for line in txt_path.read_text().splitlines():
        m = re.match(r"CDR:\s*(\S*)\s*$", line)
        if m:
            return float(m.group(1)) if m.group(1) else None
    return None


def load_oasis1_axial_slices(img_path: Path) -> list[np.ndarray]:
    """SUBJ_111 ANALYZE volume -> N_SLICES normalised uint8 axial slices."""
    import nibabel as nib

    data = np.asarray(nib.as_closest_canonical(nib.load(str(img_path))).dataobj,
                      dtype=np.float32)
    while data.ndim > 3:
        data = data[..., 0]
    n = data.shape[1]  # true axial axis (verified visually)
    lo_i = int(n * preprocess.LO_FRAC)
    hi_i = int(n * preprocess.HI_FRAC) - 1
    out = []
    for idx in np.linspace(lo_i, hi_i, preprocess.N_SLICES).astype(int):
        sl = data[:, int(idx), :]  # already anterior-up, no transform needed
        lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
        sl = np.clip((sl - lo) / (hi - lo + 1e-8), 0.0, 1.0) * 255.0
        out.append(sl.astype(np.uint8))
    return out


def predict_slices(slices: list[np.ndarray]) -> float:
    """Impaired-class probability, matching the deployed TTA/aggregation."""
    x = mri_model._to_model_input(slices)
    run = mri_model._get_session().run
    out_a = run(None, {"input": x})
    out_b = run(None, {"input": x[:, :, :, ::-1].copy()})

    def sm(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    probs = ((sm(out_a[0]) + sm(out_b[0])) / 2).mean(axis=0)
    return float(probs[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oasis1-dir", required=True)
    ap.add_argument("--out-csv", default=str(REPO / "docs" / "external_validation_oasis1.csv"))
    args = ap.parse_args()

    rows = []
    subj_dirs = sorted(Path(args.oasis1_dir).expanduser().glob("OAS1_*_MR1"))
    print(f"{len(subj_dirs)} OASIS-1 sessions found")
    for d in subj_dirs:
        txts = list(d.glob("OAS1_*.txt"))
        imgs = glob.glob(str(d / "PROCESSED" / "MPRAGE" / "SUBJ_111" / "*_sbj_111.img"))
        if not txts or not imgs:
            print(f"skip {d.name}: missing txt/img")
            continue
        cdr = read_cdr(txts[0])
        try:
            p = predict_slices(load_oasis1_axial_slices(Path(imgs[0])))
        except Exception as exc:
            print(f"FAIL {d.name}: {exc}")
            continue
        label = None if cdr is None else (0 if cdr == 0 else 1)
        rows.append({"session": d.name, "cdr": cdr, "label": label,
                     "p_impaired": round(p, 4), "pred": int(p >= 0.5)})
        print(f"{d.name}: CDR={cdr} p_impaired={p:.3f}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["session", "cdr", "label", "p_impaired", "pred"])
        w.writeheader()
        w.writerows(rows)

    labeled = [r for r in rows if r["label"] is not None]
    young = [r for r in rows if r["label"] is None]
    y = np.array([r["label"] for r in labeled])
    yhat = np.array([r["pred"] for r in labeled])
    p = np.array([r["p_impaired"] for r in labeled])

    from sklearn.metrics import confusion_matrix, roc_auc_score

    acc = float((y == yhat).mean())
    auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    young_fp = sum(r["pred"] for r in young)

    print("\n===== OASIS-1 external validation (labeled subjects) =====")
    print(f"n={len(labeled)}  (normal {int((y==0).sum())} / impaired {int((y==1).sum())})")
    print(f"accuracy={acc:.3f}  AUC={auc:.3f}  sensitivity={sens:.3f}  specificity={spec:.3f}")
    print(f"confusion [[tn fp][fn tp]] = {cm.tolist()}")
    print(f"young controls (no CDR, presumed healthy): {len(young)}, flagged impaired: {young_fp}")
    print(f"per-scan CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
