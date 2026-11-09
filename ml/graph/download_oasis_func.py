"""Download OASIS-3 resting-state BOLD (and DWI) for the graph model.

    export OASIS_USER=your_nitrc_login OASIS_PASS=your_nitrc_password
    python ml/graph/build_session_list.py   --data-root ~/Downloads/7th_sem/PP-2/data
    python ml/graph/download_oasis_func.py  --data-root ~/Downloads/7th_sem/PP-2/data --dry-run
    python ml/graph/download_oasis_func.py  --data-root ~/Downloads/7th_sem/PP-2/data --limit 3 --no-dwi
    python ml/graph/download_oasis_func.py  --data-root ~/Downloads/7th_sem/PP-2/data --no-dwi

Session list: processed/graph_sessions.csv from build_session_list.py (every
OASIS-3 session with a resting-state run and a nearby CDR, with scan ids). If
it doesn't exist yet, falls back to processed/download_list.csv (the T1
sessions chosen for the MRI work) and asks XNAT which scans each one has.

Writes oasis3_func/<experiment>/... (BOLD *.nii.gz + JSON sidecars; DWI
*.nii.gz + .bval/.bvec). Expect roughly 50-150 MB per BOLD session and
100-300 MB per DWI, so tens of GB for the full list; --no-dwi skips DWI when
only functional connectivity is wanted (the model trains on FC alone).

Server: NITRC-IR via ml/oasis_xnat.py. Safe to interrupt and re-run: finished
sessions are skipped; the script exits non-zero while anything failed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/

KEEP_SUFFIXES = (".nii.gz", ".json", ".bval", ".bvec")
MB_PER_BOLD, MB_PER_DWI = 100, 200  # rough planning numbers for --dry-run


def has_bold(session_dir: Path) -> bool:
    return bool(glob.glob(str(session_dir / "**" / "*bold.nii.gz"), recursive=True))


def has_dwi(session_dir: Path) -> bool:
    return bool(glob.glob(str(session_dir / "**" / "*dwi.nii.gz"), recursive=True))


def read_sessions(processed: Path) -> tuple[list[dict], str]:
    graph_list = processed / "graph_sessions.csv"
    if graph_list.exists():
        return list(csv.DictReader(open(graph_list))), graph_list.name
    rows = [{"experiment_id": r["experiment_id"], "subject_id": r["subject_id"],
             "bold_ids": None, "dwi_ids": None}
            for r in csv.DictReader(open(processed / "download_list.csv"))]
    return rows, "download_list.csv (run build_session_list.py for the full graph list)"


def main() -> None:
    from graph.build_session_list import is_dwi, is_rest_bold
    from oasis_xnat import AccessDenied, OasisXnat

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after N downloads (0 = all)")
    ap.add_argument("--no-dwi", action="store_true", help="resting-state BOLD only")
    ap.add_argument("--dry-run", action="store_true", help="list what would download, then stop")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    out_root = root / "oasis3_func"
    out_root.mkdir(exist_ok=True)
    sessions, source = read_sessions(root / "processed")

    todo = [s for s in sessions
            if not has_bold(out_root / s["experiment_id"])
            or (not args.no_dwi and s.get("dwi_ids") != "" and not has_dwi(out_root / s["experiment_id"]))]
    n_dwi = 0 if args.no_dwi else sum(bool(s.get("dwi_ids")) for s in todo)
    print(f"{len(todo)} of {len(sessions)} sessions still to download (from {source})")
    print(f"rough size: {len(todo) * MB_PER_BOLD / 1024:.0f} GB BOLD"
          + ("" if args.no_dwi else f" + {n_dwi * MB_PER_DWI / 1024:.0f} GB DWI"))
    if args.dry_run:
        for s in todo[:10]:
            print("  ", s["experiment_id"])
        print("   ..." if len(todo) > 10 else "", "\n(dry run: nothing downloaded)")
        return

    x = OasisXnat()
    x.login()
    done = skipped = failed = 0
    for s in todo:
        if args.limit and done >= args.limit:
            break
        exp, subject = s["experiment_id"], s["subject_id"]
        try:
            if s.get("bold_ids") is None:  # fallback list: ask which scans exist
                scans = x.scans(subject, exp)
                bold = [r["ID"] for r in scans if is_rest_bold(r.get("type"))]
                dwi = [r["ID"] for r in scans if is_dwi(r.get("type"))]
            else:
                bold, dwi = s["bold_ids"].split(), (s.get("dwi_ids") or "").split()
            ids = bold + ([] if args.no_dwi else dwi)
            if not bold:
                print(f"SKIP {exp}: no resting-state BOLD in this session")
                skipped += 1
                continue
            n = x.download_scans(subject, exp, ids, out_root, KEEP_SUFFIXES)
            done += 1
            print(f"OK   {exp}: {n} files ({done}/{len(todo)})")
        except AccessDenied as exc:
            sys.exit(str(exc))
        except Exception as exc:
            failed += 1
            print(f"FAIL {exp}: {exc}")

    print(f"\ndownloaded {done}, skipped (no rs-fMRI) {skipped}, failed {failed}")
    print("Next: python ml/graph/preprocess.py --data-root", root)
    if failed:
        sys.exit(1)  # incomplete run: re-run until it exits 0


if __name__ == "__main__":
    main()
