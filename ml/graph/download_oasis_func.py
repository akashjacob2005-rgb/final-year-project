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


def select_sessions(sessions: list[dict], labels: dict[str, dict], one_per_subject: bool,
                    max_normal: int, seed: int = 42) -> list[dict]:
    """Thin the list before downloading.

    one_per_subject: keep each person's session whose CDR was assessed closest
      to the scan (extra sessions of the same person add disk, not people).
    max_normal: keep at most this many Normal (CDR 0) people, chosen at random
      with a fixed seed; every MCI and Dementia person is kept. OASIS-3 is
      ~80% Normal, so this saves most of the download without losing any
      impaired example.
    """
    import random

    rows = [s for s in sessions if s["experiment_id"] in labels]
    if one_per_subject:
        best: dict[str, dict] = {}
        for s in rows:
            cur = best.get(s["subject_id"])
            key = float(labels[s["experiment_id"]].get("days_apart") or 0)
            if cur is None or key < float(labels[cur["experiment_id"]].get("days_apart") or 0):
                best[s["subject_id"]] = s
        rows = list(best.values())
    if max_normal:
        normal_subjects = sorted({s["subject_id"] for s in rows if labels[s["experiment_id"]]["label"] == "0"})
        if len(normal_subjects) > max_normal:
            keep = set(random.Random(seed).sample(normal_subjects, max_normal))
            rows = [s for s in rows if labels[s["experiment_id"]]["label"] != "0" or s["subject_id"] in keep]
    return sorted(rows, key=lambda s: s["experiment_id"])


def main() -> None:
    from graph.build_session_list import is_dwi, is_rest_bold
    from oasis_xnat import AccessDenied, OasisXnat

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after N downloads (0 = all)")
    ap.add_argument("--no-dwi", action="store_true", help="resting-state BOLD only")
    ap.add_argument("--dry-run", action="store_true", help="list what would download, then stop")
    ap.add_argument("--one-per-subject", action="store_true",
                    help="one session per person (the one with the closest CDR)")
    ap.add_argument("--max-normal", type=int, default=0,
                    help="keep at most N Normal people (all MCI/Dementia kept); 0 = no cap")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    out_root = root / "oasis3_func"
    out_root.mkdir(exist_ok=True)
    sessions, source = read_sessions(root / "processed")
    label_file = root / "processed" / "graph_session_labels.csv"
    if (args.one_per_subject or args.max_normal) and label_file.exists():
        labels = {r["mr_session"]: r for r in csv.DictReader(open(label_file))}
        sessions = select_sessions(sessions, labels, args.one_per_subject, args.max_normal)
        counts = {c: sum(labels[s["experiment_id"]]["label"] == c for s in sessions) for c in "012"}
        print(f"selection: {len(sessions)} sessions — Normal {counts['0']} · MCI {counts['1']} · "
              f"Dementia {counts['2']}")

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
