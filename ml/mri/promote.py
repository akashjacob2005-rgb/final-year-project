"""Promote a finished experiment to the deployed MRI artifact.

This is the ONLY code path that writes ml/artifacts/mri_model.onnx(.data) and
ml/artifacts/mri_metrics.json. train_mri.py --experiment never touches them,
so a candidate can only reach production by passing these checks:

  1. ml/experiments/<name>/final/{model.onnx,model.onnx.data,mri_metrics.json}
     exist (train_mri.py --final-fit) — the model trained on ALL subjects.
  2. The metrics carry the keys the backend and frontend read
     (classes, cv.{folds,accuracy,f1,auc}.{mean,std}).
  3. The ONNX graph has input 'input' and outputs (logits, cam), and agrees
     with the torch checkpoint on a synthetic batch (max |dp| < 1e-3) when
     torch is importable.
  4. The backend's predict() runs on a synthetic NIfTI volume against the
     candidate and returns N_SLICES heatmaps.
  5. <name>/calibration.json (compare_experiments.py --calibrate) is merged in
     as metrics["calibration"]; without it the backend keeps plain argmax and
     the script says so loudly.

Then it copies the files, records provenance in metrics["promoted_from"] and
reminds you what to run next (backend tests, external validation, model card).
Nothing is written without --yes.

Usage (from repo root, backend venv so onnxruntime/nibabel are importable):
    backend/.venv/bin/python ml/mri/promote.py e04_swa --yes
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "ml" / "artifacts"
EXPERIMENTS = REPO / "ml" / "experiments"
REQUIRED_CV = ("folds", "accuracy", "f1", "auc")


def fail(msg: str) -> None:
    raise SystemExit(f"REFUSED: {msg}")


def check_metrics(metrics: dict) -> None:
    if not metrics.get("classes"):
        fail("mri_metrics.json has no 'classes'")
    cv = metrics.get("cv") or {}
    for k in REQUIRED_CV:
        if k not in cv:
            fail(f"mri_metrics.json cv block lacks '{k}' (frontend reads cv.{k})")
    for k in ("accuracy", "f1", "auc"):
        if not {"mean", "std"} <= set(cv[k]):
            fail(f"cv.{k} must have mean and std")


def check_onnx(onnx_path: Path, n_classes: int, ckpt_path: Path | None) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names_in = [i.name for i in sess.get_inputs()]
    names_out = [o.name for o in sess.get_outputs()]
    if names_in != ["input"]:
        fail(f"ONNX inputs are {names_in}, backend expects ['input']")
    if names_out[:2] != ["logits", "cam"]:
        fail(f"ONNX outputs are {names_out}, backend expects ['logits', 'cam']")
    x = np.random.default_rng(0).standard_normal((4, 3, 224, 224)).astype(np.float32)
    logits, cam = sess.run(None, {"input": x})[:2]
    if logits.shape != (4, n_classes):
        fail(f"logits shape {logits.shape}, expected (4, {n_classes}) — dynamic batch or class count broken")
    if cam.shape[:2] != (4, n_classes):
        fail(f"cam shape {cam.shape}, expected (4, {n_classes}, H, W)")
    print(f"ok  ONNX graph: input 'input', outputs {names_out}, logits {logits.shape}, cam {cam.shape}")

    if ckpt_path is None or not ckpt_path.exists():
        print("--  torch checkpoint absent; skipping ONNX-vs-torch agreement check")
        return
    try:
        import torch

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from export_onnx import load_resnet
    except Exception as exc:  # torch not installed in this venv
        print(f"--  torch unavailable ({type(exc).__name__}); skipping ONNX-vs-torch agreement check")
        return
    net = load_resnet(ckpt_path)
    with torch.no_grad():
        ref = net(torch.from_numpy(x)).numpy()
    diff = float(np.abs(ref - logits).max())
    if diff > 1e-3:
        fail(f"ONNX logits differ from torch by {diff:.2e} (> 1e-3)")
    print(f"ok  ONNX matches torch checkpoint (max |dlogit| = {diff:.1e})")


def check_backend(onnx_path: Path, metrics_path: Path) -> dict:
    """Run the real backend predict() against the candidate on a synthetic volume."""
    import gzip

    sys.path.insert(0, str(REPO / "backend"))
    try:
        from app.ml import mri_model
    except Exception as exc:
        fail(f"cannot import backend (run from backend/.venv): {exc}")
    import nibabel as nib
    from mri import preprocess

    mri_model.MRI_MODEL_PATH = onnx_path
    mri_model.MRI_METRICS_PATH = metrics_path
    mri_model.reset()
    rng = np.random.default_rng(0)
    img = nib.Nifti1Image((rng.random((64, 64, 64)) * 1000).astype(np.float32), affine=np.eye(4))
    out = mri_model.predict(gzip.compress(img.to_bytes()))
    if out["slices_analysed"] != preprocess.N_SLICES or len(out["slices"]) != preprocess.N_SLICES:
        fail(f"backend returned {len(out['slices'])} heatmaps for {out['slices_analysed']} slices")
    if abs(sum(out["probabilities"].values()) - 1.0) > 1e-3:
        fail("probabilities do not sum to 1")
    print(f"ok  backend predict(): {out['predicted_class']!r} conf {out['confidence']:.3f} "
          f"calibrated={out['calibrated']} threshold={out['decision_threshold']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="name under ml/experiments/ (must have final/ from --final-fit)")
    ap.add_argument("--experiments-root", default=str(EXPERIMENTS))
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    ap.add_argument("--yes", action="store_true", help="actually write ml/artifacts (default: dry run)")
    ap.add_argument("--allow-uncalibrated", action="store_true",
                    help="promote without calibration.json (plain argmax decisions)")
    args = ap.parse_args()

    exp = Path(args.experiments_root) / args.experiment
    final = exp / "final"
    artifacts = Path(args.artifacts)
    onnx_path, data_path = final / "model.onnx", final / "model.onnx.data"
    metrics_path, ckpt_path = final / "mri_metrics.json", final / "model.pt"
    for p in (onnx_path, data_path, metrics_path):
        if not p.exists():
            fail(f"{p} missing — run train_mri.py --experiment {args.experiment} ... --final-fit")

    metrics = json.loads(metrics_path.read_text())
    check_metrics(metrics)
    summary = json.loads((exp / "summary.json").read_text()) if (exp / "summary.json").exists() else {}
    cv = metrics["cv"]
    print(f"candidate {args.experiment}: {metrics['model']}")
    print(f"    cv acc {cv['accuracy']['mean']:.3f}±{cv['accuracy']['std']:.3f}  "
          f"auc {cv['auc']['mean']:.3f}±{cv['auc']['std']:.3f}  "
          f"(folds {cv['folds']}, repeats {cv.get('repeats', 1)}); {metrics['deployed']['which']}")

    cal_path = exp / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        metrics["calibration"] = cal
        est = cal.get("cross_fitted_estimate", {})
        print(f"ok  calibration: T={cal['temperature']} threshold={cal['threshold']} "
              f"(cross-fitted bal acc {est.get('balanced_accuracy', {}).get('mean', '?')}, "
              f"normal recall {est.get('recall_normal', {}).get('mean', '?')})")
    elif args.allow_uncalibrated:
        metrics.pop("calibration", None)
        print("!!  no calibration.json — deploying with plain argmax (--allow-uncalibrated)")
    else:
        fail(f"{cal_path} missing — run compare_experiments.py --calibrate {args.experiment} "
             "(or pass --allow-uncalibrated)")

    # Compare with what is deployed now, so the swap is a conscious decision.
    current = artifacts / "mri_metrics.json"
    if current.exists():
        cur = json.loads(current.read_text())
        ccv = cur.get("cv", {})
        print(f"currently deployed: acc {ccv.get('accuracy', {}).get('mean', '?')} "
              f"auc {ccv.get('auc', {}).get('mean', '?')} (version {cur.get('version', '?')})")

    check_onnx(onnx_path, len(metrics["classes"]), ckpt_path)
    # Write the merged metrics next to the candidate so the backend check sees
    # the calibration block exactly as it will be deployed.
    staged = final / "mri_metrics.promote.json"
    metrics["promoted_from"] = {
        "experiment": args.experiment,
        "git_sha": summary.get("git_sha") or _config_sha(exp),
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": _config_argv(exp),
    }
    metrics["version"] = f"{metrics.get('version', '2.1.0')}+{args.experiment}"
    staged.write_text(json.dumps(metrics, indent=2))
    check_backend(onnx_path, staged)

    if not args.yes:
        staged.unlink()
        print("\nDRY RUN — all checks passed. Re-run with --yes to write ml/artifacts/.")
        return

    artifacts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, artifacts / "mri_model.onnx")
    shutil.copy2(data_path, artifacts / "mri_model.onnx.data")
    shutil.move(str(staged), artifacts / "mri_metrics.json")
    if ckpt_path.exists():
        shutil.copy2(ckpt_path, artifacts / "mri_model.pt")  # git-ignored; keeps export_onnx.py reproducible
    print(f"\npromoted {args.experiment} -> {artifacts}/mri_model.onnx(.data), mri_metrics.json")
    print("next:  cd backend && .venv/bin/python -m pytest tests/ -q")
    print("       backend/.venv/bin/python ml/mri/external_validate.py --oasis1-dir <discs 1-2>   # once, report only")
    print("       update docs/MODEL_CARD.md + docs/ARCH_COMPARISON.md, then commit (push only with approval)")


def _config_sha(exp: Path) -> str | None:
    cfg = exp / "config.json"
    return json.loads(cfg.read_text()).get("git_sha") if cfg.exists() else None


def _config_argv(exp: Path) -> str | None:
    cfg = exp / "config.json"
    return " ".join(json.loads(cfg.read_text()).get("argv", [])) if cfg.exists() else None


if __name__ == "__main__":
    main()
