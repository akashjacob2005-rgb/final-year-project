"""Promote finished experiments to the deployed MRI artifacts.

This is the ONLY code path that writes the MRI files in ml/artifacts/.
train_mri.py --experiment never touches them, so a candidate reaches the app
only by passing these checks.

A deployment is one ResNet-18 member (mandatory: it carries the CAM heatmap
output) plus optional transformer members, each the `final/` folder of an
experiment trained with --final-fit:

    promote.py e04_resnet_swa                                   # single model
    promote.py e04_resnet_swa e07_vit e08_dinov2 \\
        --calibration-from ens_e04_e07_e08                      # 3-model ensemble

Steps:
  1. Stage a bundle laid out exactly like ml/artifacts in
     ml/experiments/_staged/: mri_model.onnx(.data) [+ mri_model_<arch>.onnx],
     mri_metrics.json [+ mri_ensemble_metrics.json for an ensemble].
  2. Merge calibration.json (compare_experiments.py --calibrate) into the
     metrics file the backend reads for that member set: mri_metrics.json for
     a single model, mri_ensemble_metrics.json for an ensemble. The backend
     ignores a calibration fitted for a different member set.
  3. Check every member's ONNX I/O, then run the real backend predict() on a
     synthetic volume against the staged bundle.
  4. With --yes: copy the bundle into ml/artifacts (removing member files of
     archs that are no longer part of the deployment). Without: dry run.

--calibrate-deployed EXP skips all of the above and only installs EXP's
calibration.json into the CURRENTLY deployed artifacts (the quick win: the
existing ensemble's out-of-fold predictions are enough, no retraining):

    promote.py --calibrate-deployed ens_sep27 --yes

--stage-only stops after step 3 so external_validate.py --model-dir can test
the bundle on OASIS-1 discs 1-2 BEFORE anything is deployed.

Usage (from repo root, backend venv so onnxruntime/nibabel are importable):
    backend/.venv/bin/python ml/mri/promote.py EXP [EXP ...] [--calibration-from EXP] [--yes]
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
ARCH_FILES = {"resnet18": "mri_model", "dinov2_s": "mri_model_dinov2_s", "vit_b16": "mri_model_vit_b16"}


def fail(msg: str) -> None:
    raise SystemExit(f"REFUSED: {msg}")


def check_metrics(metrics: dict, where: str) -> None:
    if not metrics.get("classes") and "members" not in metrics:
        fail(f"{where} has no 'classes'")
    cv = metrics.get("cv", metrics)  # ensemble metrics keep cv keys at top level
    for k in REQUIRED_CV:
        if k not in cv:
            fail(f"{where} lacks '{k}' (the frontend reads it)")
    for k in ("accuracy", "f1", "auc"):
        if not {"mean", "std"} <= set(cv[k]):
            fail(f"{where}: {k} must have mean and std")


def check_onnx(onnx_path: Path, n_classes: int, needs_cam: bool) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names_in = [i.name for i in sess.get_inputs()]
    names_out = [o.name for o in sess.get_outputs()]
    if names_in != ["input"]:
        fail(f"{onnx_path.name}: inputs {names_in}, backend expects ['input']")
    if names_out[0] != "logits" or (needs_cam and names_out[:2] != ["logits", "cam"]):
        fail(f"{onnx_path.name}: outputs {names_out}, expected logits{' + cam' if needs_cam else ''}")
    x = np.random.default_rng(0).standard_normal((4, 3, 224, 224)).astype(np.float32)
    out = sess.run(None, {"input": x})
    if out[0].shape != (4, n_classes):
        fail(f"{onnx_path.name}: logits shape {out[0].shape}, expected (4, {n_classes})")
    print(f"ok  {onnx_path.name}: outputs {names_out}, batch-dynamic")


def check_backend(bundle: Path) -> dict:
    """Run the real backend predict() against the staged bundle."""
    import gzip

    sys.path.insert(0, str(REPO / "backend"))
    try:
        from app.ml import mri_model
    except Exception as exc:
        fail(f"cannot import backend (run with backend/.venv): {exc}")
    import nibabel as nib
    from mri import preprocess

    mri_model.MRI_MODEL_PATH = bundle / "mri_model.onnx"
    mri_model.MRI_METRICS_PATH = bundle / "mri_metrics.json"
    mri_model.MRI_ENSEMBLE_METRICS_PATH = bundle / "mri_ensemble_metrics.json"
    mri_model.reset()
    rng = np.random.default_rng(0)
    img = nib.Nifti1Image((rng.random((64, 64, 64)) * 1000).astype(np.float32), affine=np.eye(4))
    out = mri_model.predict(gzip.compress(img.to_bytes()))
    if out["slices_analysed"] != preprocess.N_SLICES or len(out["slices"]) != preprocess.N_SLICES:
        fail(f"backend returned {len(out['slices'])} heatmaps for {out['slices_analysed']} slices")
    if abs(sum(out["probabilities"].values()) - 1.0) > 1e-3:
        fail("probabilities do not sum to 1")
    print(f"ok  backend predict(): members {out['model_members']}, {out['predicted_class']!r} "
          f"conf {out['confidence']:.3f}, calibrated={out['calibrated']}, threshold={out['decision_threshold']}")
    mri_model.reset()
    return out


def stage(experiments: list[str], root: Path, cal_from: str | None, allow_uncal: bool) -> tuple[Path, list[str]]:
    staged = root / "_staged"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    archs: dict[str, Path] = {}
    for name in experiments:
        final = root / name / "final"
        metrics_files = sorted(final.glob("mri_metrics*.json"))
        if not metrics_files:
            fail(f"{final} has no mri_metrics*.json — train with --experiment {name} ... --final-fit")
        for arch, stem in ARCH_FILES.items():
            if (final / f"{stem}.onnx").exists():
                if arch in archs:
                    fail(f"two experiments provide {arch}: {archs[arch].parent.name} and {name}")
                archs[arch] = final
    if "resnet18" not in archs:
        fail("no ResNet-18 member: it is mandatory (it produces the heatmaps)")
    members = [a for a in ARCH_FILES if a in archs]

    for arch in members:
        stem, final = ARCH_FILES[arch], archs[arch]
        for f in (f"{stem}.onnx", f"{stem}.onnx.data"):
            if (final / f).exists():
                shutil.copy2(final / f, staged / f)
        if not (staged / f"{stem}.onnx.data").exists() and arch == "resnet18":
            print("--  no external-data sidecar for the ResNet (fine for small graphs)")

    resnet_metrics = json.loads((archs["resnet18"] / "mri_metrics.json").read_text())
    check_metrics(resnet_metrics, "resnet18 final/mri_metrics.json")

    cal = None
    cal_name = cal_from or (experiments[0] if len(members) == 1 else None)
    if cal_name:
        cal_path = root / cal_name / "calibration.json"
        if cal_path.exists():
            cal = json.loads(cal_path.read_text())
            if sorted(cal.get("members", ["resnet18"])) != sorted(members):
                fail(f"{cal_path} was fitted for {cal.get('members')} but deploying {members}")
    if cal is None and not allow_uncal:
        fail("no calibration: run compare_experiments.py --calibrate on "
             + ("the member experiment" if len(members) == 1 else
                "the ensemble (ensemble_eval.py ... --write-preds ml/experiments/<ens>), then --calibration-from <ens>")
             + " — or pass --allow-uncalibrated")

    provenance = {"experiments": experiments, "members": members,
                  "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    resnet_metrics["promoted_from"] = provenance
    resnet_metrics.pop("calibration", None)
    if len(members) == 1:
        if cal:
            resnet_metrics["calibration"] = cal
    else:
        ens_dir = root / cal_name if cal_name else None
        ens_summary = json.loads((ens_dir / "summary.json").read_text()) if ens_dir and (ens_dir / "summary.json").exists() else {}
        ens = {"members": members, "folds": resnet_metrics["cv"]["folds"],
               **{k: v for k, v in ens_summary.get("cv", {}).items()},
               "promoted_from": provenance}
        if cal:
            ens["calibration"] = cal
        check_metrics(ens, "ensemble metrics")
        (staged / "mri_ensemble_metrics.json").write_text(json.dumps(ens, indent=2))
    (staged / "mri_metrics.json").write_text(json.dumps(resnet_metrics, indent=2))

    n_classes = len(resnet_metrics["classes"])
    for arch in members:
        check_onnx(staged / f"{ARCH_FILES[arch]}.onnx", n_classes, needs_cam=arch == "resnet18")
    if cal:
        est = cal.get("cross_fitted_estimate", {})
        print(f"ok  calibration for {members}: T={cal['temperature']} threshold={cal['threshold']} "
              f"(cross-fitted bal acc {est.get('balanced_accuracy', {}).get('mean', '?')}, "
              f"normal recall {est.get('recall_normal', {}).get('mean', '?')})")
    return staged, members


def calibrate_deployed(exp: Path, artifacts: Path, yes: bool) -> None:
    """Install a calibration into the deployed metrics file for its member set."""
    cal_path = exp / "calibration.json"
    if not cal_path.exists():
        fail(f"{cal_path} missing — run compare_experiments.py {exp.name} --calibrate {exp.name}")
    cal = json.loads(cal_path.read_text())
    deployed = [a for a, stem in ARCH_FILES.items() if (artifacts / f"{stem}.onnx").exists()]
    if sorted(cal.get("members", ["resnet18"])) != sorted(deployed):
        fail(f"calibration fitted for {cal.get('members')} but deployed members are {deployed}")
    target = artifacts / ("mri_ensemble_metrics.json" if len(deployed) > 1 else "mri_metrics.json")
    if not target.exists():
        fail(f"{target} missing")
    metrics = json.loads(target.read_text())
    metrics["calibration"] = cal
    est = cal.get("cross_fitted_estimate", {})
    print(f"calibration for {deployed}: T={cal['temperature']} threshold={cal['threshold']} -> {target.name}")
    print(f"    honest (cross-fitted) estimate: bal acc {est.get('balanced_accuracy', {}).get('mean')}, "
          f"normal recall {est.get('recall_normal', {}).get('mean')}, impaired recall {est.get('recall_impaired', {}).get('mean')}")
    if not yes:
        print("DRY RUN — re-run with --yes to write it.")
        return
    target.write_text(json.dumps(metrics, indent=2))
    print(f"written {target}; restart the backend to pick it up.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", help="experiment names with final/ (one per member arch)")
    ap.add_argument("--calibrate-deployed", default=None, metavar="EXP",
                    help="only install EXP/calibration.json into the deployed artifacts")
    ap.add_argument("--calibration-from", default=None,
                    help="experiment dir holding calibration.json for this member set "
                         "(default: the single member's own experiment)")
    ap.add_argument("--experiments-root", default=str(EXPERIMENTS))
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    ap.add_argument("--allow-uncalibrated", action="store_true")
    ap.add_argument("--stage-only", action="store_true",
                    help="build + check ml/experiments/_staged and stop (for external_validate.py --model-dir)")
    ap.add_argument("--yes", action="store_true", help="actually write ml/artifacts (default: dry run)")
    args = ap.parse_args()

    root = Path(args.experiments_root)
    if args.calibrate_deployed:
        calibrate_deployed(root / args.calibrate_deployed, Path(args.artifacts), args.yes)
        return
    if not args.experiments:
        ap.error("give experiment names, or --calibrate-deployed EXP")
    staged, members = stage(args.experiments, root, args.calibration_from, args.allow_uncalibrated)

    current = Path(args.artifacts) / "mri_metrics.json"
    if current.exists():
        ccv = json.loads(current.read_text()).get("cv", {})
        print(f"currently deployed ResNet: acc {ccv.get('accuracy', {}).get('mean', '?')} "
              f"auc {ccv.get('auc', {}).get('mean', '?')}")

    check_backend(staged)
    if args.stage_only:
        print(f"\nSTAGED (not deployed): {staged}\n"
              f"external test: backend/.venv/bin/python ml/mri/external_validate.py "
              f"--oasis1-dir <discs 1-2> --model-dir {staged}")
        return
    if not args.yes:
        print("\nDRY RUN — all checks passed. Re-run with --yes to write ml/artifacts/.")
        return

    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    for arch, stem in ARCH_FILES.items():
        if arch not in members:
            for f in (artifacts / f"{stem}.onnx", artifacts / f"{stem}.onnx.data"):
                if f.exists():
                    f.unlink()
                    print(f"removed {f.name} ({arch} is not part of this deployment)")
    if len(members) == 1 and (artifacts / "mri_ensemble_metrics.json").exists():
        (artifacts / "mri_ensemble_metrics.json").unlink()
    for f in staged.iterdir():
        shutil.copy2(f, artifacts / f.name)
    print(f"\npromoted {members} from {args.experiments} -> {artifacts}")
    print("next:  cd backend && .venv/bin/python -m pytest tests/ -q")
    print("       update docs/MODEL_CARD.md + docs/MRI_EXPERIMENTS.md with the numbers; commit (push only with approval)")


if __name__ == "__main__":
    main()
