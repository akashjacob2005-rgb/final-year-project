"""Download the missing OASIS-3 T1w sessions listed in download_list.csv.

OASIS-3 is served from NITRC-IR (XNAT) and needs a NITRC account granted the
OASIS3 project (https://sites.wustl.edu/oasisbrains/ -> apply for access).
Credentials are read from the environment and never written to disk:

    export OASIS_USER=your_nitrc_login
    export OASIS_PASS=your_nitrc_password
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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/ for oasis_xnat


def has_t1w(session_dir: Path) -> bool:
    return bool(glob.glob(str(session_dir / "**" / "*T1w.nii.gz"), recursive=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dir containing oasis3_nii/ and processed/download_list.csv")
    ap.add_argument("--limit", type=int, default=0, help="stop after N downloads (0 = all)")
    args = ap.parse_args()

    from oasis_xnat import AccessDenied, OasisXnat

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

    x = OasisXnat()
    x.login()

    done = failed = 0
    for exp, subject in todo:
        if args.limit and done >= args.limit:
            break
        try:
            t1w_ids = [r["ID"] for r in x.scans(subject, exp) if "T1w" in (r.get("type") or "")]
            if not t1w_ids:
                print(f"SKIP {exp}: no T1w scans listed")
                failed += 1
                continue
            x.download_scans(subject, exp, t1w_ids, nii_root, (".nii.gz", "T1w.json"))
            done += 1
            print(f"OK   {exp} ({done}/{len(todo)})")
        except AccessDenied as exc:
            sys.exit(str(exc))
        except Exception as exc:
            failed += 1
            print(f"FAIL {exp}: {exc}")

    print(f"\ndownloaded {done}, failed/skipped {failed}")
    print("Next: python ml/mri/make_slices.py --data-root", root)


if __name__ == "__main__":
    main()
