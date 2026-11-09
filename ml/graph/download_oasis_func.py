"""Download OASIS-3 resting-state BOLD and DWI for the sessions in download_list.csv.

Mirrors ml/mri/download_oasis.py (same XNAT login, same session list, same
"skip what is already on disk" behaviour). OASIS-3 needs approved credentials
(https://sites.wustl.edu/oasisbrains/ -> apply for access), passed only via
environment variables:

    export OASIS_USER=your_username
    export OASIS_PASS=your_password
    python ml/graph/download_oasis_func.py --data-root /path/to/data [--limit 20] [--no-dwi]

Writes oasis3_func/<experiment>/... (BOLD *.nii.gz + JSON sidecars, DWI
*.nii.gz + .bval/.bvec + JSON). A rs-fMRI session is ~50-150MB and a DWI
~100-300MB, so expect tens of GB for the full list; --no-dwi skips DWI when
only functional connectivity is wanted.

Not every OASIS-3 MR session has a resting-state run or DWI: those are
reported as SKIP and counted, never treated as failures to retry forever.

Safe to interrupt and re-run: finished sessions are skipped.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import sys
import zipfile
from pathlib import Path

BASE = "https://central.xnat.org"
PROJECT = "OASIS3"
KEEP_SUFFIXES = (".nii.gz", ".json", ".bval", ".bvec")


def is_rest_bold(scan_type: str) -> bool:
    t = scan_type.lower()
    return "bold" in t and "task" not in t.replace("task-rest", "")


def is_dwi(scan_type: str) -> bool:
    return "dwi" in scan_type.lower() or "dti" in scan_type.lower()


def has_bold(session_dir: Path) -> bool:
    return bool(glob.glob(str(session_dir / "**" / "*bold.nii.gz"), recursive=True))


def has_dwi(session_dir: Path) -> bool:
    return bool(glob.glob(str(session_dir / "**" / "*dwi.nii.gz"), recursive=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dir containing processed/download_list.csv")
    ap.add_argument("--limit", type=int, default=0, help="stop after N downloads (0 = all)")
    ap.add_argument("--no-dwi", action="store_true", help="resting-state BOLD only")
    args = ap.parse_args()

    import requests

    user, password = os.getenv("OASIS_USER"), os.getenv("OASIS_PASS")
    if not user or not password:
        sys.exit("Set OASIS_USER and OASIS_PASS environment variables first.")

    root = Path(args.data_root)
    out_root = root / "oasis3_func"
    out_root.mkdir(exist_ok=True)

    todo = []
    with open(root / "processed" / "download_list.csv") as f:
        for row in csv.DictReader(f):
            exp = row["experiment_id"]
            need_dwi = not args.no_dwi and not has_dwi(out_root / exp)
            if not has_bold(out_root / exp) or need_dwi:
                todo.append((exp, row["subject_id"]))
    print(f"{len(todo)} sessions still to download")

    s = requests.Session()
    s.auth = (user, password)
    r = s.post(f"{BASE}/data/JSESSION", timeout=30)
    if r.status_code != 200:
        sys.exit(f"XNAT login failed (HTTP {r.status_code}) — check the credentials.")

    done = skipped = failed = 0
    for exp, subject in todo:
        if args.limit and done >= args.limit:
            break
        try:
            scans_url = (
                f"{BASE}/data/archive/projects/{PROJECT}/subjects/{subject}"
                f"/experiments/{exp}/scans?format=csv"
            )
            rows = list(csv.DictReader(io.StringIO(s.get(scans_url, timeout=60).text)))
            ids = [r["ID"] for r in rows if is_rest_bold(r.get("type") or "")]
            if not args.no_dwi:
                ids += [r["ID"] for r in rows if is_dwi(r.get("type") or "")]
            if not ids:
                print(f"SKIP {exp}: no resting-state BOLD{'' if args.no_dwi else ' or DWI'} in this session")
                skipped += 1
                continue

            files_url = (
                f"{BASE}/data/archive/projects/{PROJECT}/subjects/{subject}"
                f"/experiments/{exp}/scans/{','.join(ids)}/files?format=zip"
            )
            resp = s.get(files_url, timeout=1800)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                for m in z.namelist():
                    if not m.endswith(KEEP_SUFFIXES):
                        continue
                    target = out_root / m  # zip paths already start with the experiment label
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(m) as src, open(target, "wb") as dst:
                        dst.write(src.read())
            done += 1
            print(f"OK   {exp} ({done}/{len(todo)})")
        except Exception as exc:
            failed += 1
            print(f"FAIL {exp}: {exc}")

    print(f"\ndownloaded {done}, skipped (no rs-fMRI/DWI) {skipped}, failed {failed}")
    print("Next: python ml/graph/preprocess.py --data-root", root)
    if failed:
        sys.exit(1)  # incomplete run: re-run until it exits 0


if __name__ == "__main__":
    main()
