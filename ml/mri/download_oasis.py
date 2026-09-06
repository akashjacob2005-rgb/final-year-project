"""Download the missing OASIS-3 T1w sessions listed in download_list.csv.

OASIS-3 lives on XNAT Central and requires approved credentials
(https://sites.wustl.edu/oasisbrains/ -> apply for access). Pass them via
environment variables — they are never written to disk:

    export OASIS_USER=your_username
    export OASIS_PASS=your_password
    python ml/mri/download_oasis.py --data-root /path/to/data [--limit 20]

For every experiment in processed/download_list.csv that has no T1w volume
under oasis3_nii/ yet, the script asks XNAT which scans are T1w and downloads
only those (keeping bandwidth and disk at roughly 10-25MB per session instead
of the full multi-modal archive). Zips are extracted straight into
oasis3_nii/<experiment>/... — the same layout the existing 181 sessions use —
and deleted afterwards, so make_slices.py picks the new sessions up unchanged.

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


def has_t1w(session_dir: Path) -> bool:
    return bool(glob.glob(str(session_dir / "**" / "*T1w.nii.gz"), recursive=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dir containing oasis3_nii/ and processed/download_list.csv")
    ap.add_argument("--limit", type=int, default=0, help="stop after N downloads (0 = all)")
    args = ap.parse_args()

    import requests

    user, password = os.getenv("OASIS_USER"), os.getenv("OASIS_PASS")
    if not user or not password:
        sys.exit("Set OASIS_USER and OASIS_PASS environment variables first.")

    root = Path(args.data_root)
    nii_root = root / "oasis3_nii"
    nii_root.mkdir(exist_ok=True)

    todo = []
    with open(root / "processed" / "download_list.csv") as f:
        for row in csv.DictReader(f):
            exp = row["experiment_id"]
            if not has_t1w(nii_root / exp):
                todo.append((exp, row["subject_id"]))
    print(f"{len(todo)} sessions still to download")

    s = requests.Session()
    s.auth = (user, password)
    # A cookie-backed session avoids re-authenticating on every request.
    r = s.post(f"{BASE}/data/JSESSION", timeout=30)
    if r.status_code != 200:
        sys.exit(f"XNAT login failed (HTTP {r.status_code}) — check the credentials.")

    done = failed = 0
    for exp, subject in todo:
        if args.limit and done >= args.limit:
            break
        try:
            scans_url = (
                f"{BASE}/data/archive/projects/{PROJECT}/subjects/{subject}"
                f"/experiments/{exp}/scans?format=csv"
            )
            rows = list(csv.DictReader(io.StringIO(s.get(scans_url, timeout=60).text)))
            t1w_ids = [r["ID"] for r in rows if "T1w" in (r.get("type") or "")]
            if not t1w_ids:
                print(f"SKIP {exp}: no T1w scans listed")
                failed += 1
                continue

            files_url = (
                f"{BASE}/data/archive/projects/{PROJECT}/subjects/{subject}"
                f"/experiments/{exp}/scans/{','.join(t1w_ids)}/files?format=zip"
            )
            resp = s.get(files_url, timeout=600)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                members = [m for m in z.namelist() if m.endswith(".nii.gz") or m.endswith("T1w.json")]
                for m in members:
                    # zip paths already start with the experiment label
                    target = nii_root / m
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(m) as src, open(target, "wb") as dst:
                        dst.write(src.read())
            done += 1
            print(f"OK   {exp} ({done}/{len(todo)})")
        except Exception as exc:
            failed += 1
            print(f"FAIL {exp}: {exc}")

    print(f"\ndownloaded {done}, failed/skipped {failed}")
    print("Next: python ml/mri/make_slices.py --data-root", root)


if __name__ == "__main__":
    main()
