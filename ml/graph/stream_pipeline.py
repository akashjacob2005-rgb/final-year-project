"""Download -> preprocess -> delete, one session at a time (low-disk mode).

The raw rs-fMRI for ~900 OASIS-3 sessions is 50-90 GB, but each session ends
up as one N x N connectivity matrix of a few KB. This script never keeps the
raw data: it downloads a session's BOLD into a scratch folder, runs the same
preprocessing as ml/graph/preprocess.py (motion correction, MNI registration,
atlas FC), saves processed_graph/<session>/fc.npy, and deletes the scratch
folder. Peak disk use is about (workers + 1) sessions, i.e. under 1 GB.

    export OASIS_USER=... OASIS_PASS=...
    D=~/Downloads/7th_sem/PP-2/data
    python ml/graph/stream_pipeline.py --data-root $D --one-per-subject --max-normal 500 --limit 3
    caffeinate -i python ml/graph/stream_pipeline.py --data-root $D --one-per-subject --max-normal 500

Resumable: sessions whose fc.npy exists are skipped, so stop it any time and
re-run the same command. Sessions excluded for head motion are recorded in
processed_graph/excluded.txt and not retried. Runs BOLD only (DTI needs
tractography outside Python; see preprocess.py).

Then: python ml/graph/train_graph.py --data-dir $D/processed_graph
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/

CORES = os.cpu_count() or 2


def _limit_threads(n: int) -> None:
    """Worker initializer: ANTs/ITK otherwise uses every core in every worker,
    and N workers x all cores thrash instead of running in parallel."""
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)


def first_run_only(ids: list[str], all_runs: bool) -> list[str]:
    """One resting-state run per session is enough for an FC matrix and halves
    the download; --all-runs keeps every run (only the first is processed)."""
    return ids if all_runs else ids[:1]


def main() -> None:
    from graph.atlas import DEFAULT_ATLAS, load_atlas
    from graph.download_oasis_func import read_sessions, select_sessions
    from graph.preprocess import _one, find_bold, load_labels, write_manifest
    from oasis_xnat import AccessDenied, OasisXnat

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--atlas", default=DEFAULT_ATLAS)
    ap.add_argument("--workers", type=int, default=max(1, CORES // 2),
                    help="sessions processed in parallel (each holds one raw download on disk; "
                         "CPU cores are split between them); 0 = in this process (debugging)")
    ap.add_argument("--all-runs", action="store_true", help="download every resting-state run")
    ap.add_argument("--one-per-subject", action="store_true")
    ap.add_argument("--max-normal", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="stop after N new sessions")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    out_dir = root / "processed_graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = root / "_stream_scratch"
    shutil.rmtree(scratch, ignore_errors=True)  # leftovers from an interrupted run
    scratch.mkdir()
    excluded_file = out_dir / "excluded.txt"
    excluded = set(excluded_file.read_text().split()) if excluded_file.exists() else set()

    labels, label_name = load_labels(root)
    sessions, source = read_sessions(root / "processed")
    label_rows = {k: {"label": str(v), "days_apart": "0"} for k, v in labels.items()}
    if (root / "processed" / "graph_session_labels.csv").exists():
        label_rows = {r["mr_session"]: r for r in csv.DictReader(open(root / "processed" / "graph_session_labels.csv"))}
    sessions = select_sessions(sessions, label_rows, args.one_per_subject, args.max_normal)
    todo = [s for s in sessions if not (out_dir / s["experiment_id"] / "fc.npy").exists()
            and s["experiment_id"] not in excluded]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(sessions)} selected sessions ({source}, labels {label_name}); {len(todo)} to process")
    if not todo:
        write_manifest(out_dir, labels)
        return

    (out_dir / "regions.json").write_text(json.dumps(load_atlas(args.atlas)[1], indent=1))
    x = OasisXnat()
    x.login()

    def fetch(s) -> Path | None:
        from graph.build_session_list import is_rest_bold

        exp, subject = s["experiment_id"], s["subject_id"]
        ids = (s.get("bold_ids") or "").split() or [
            r["ID"] for r in x.scans(subject, exp) if is_rest_bold(r.get("type"))]
        if not ids:
            return None
        t0 = time.time()
        x.download_scans(subject, exp, first_run_only(ids, args.all_runs), scratch, (".nii.gz", ".json"))
        download_secs[exp] = time.time() - t0
        return find_bold(scratch / exp)

    def stage_times(sess: str) -> str:
        meta = out_dir / sess / "meta.txt"
        parts = [f"download {download_secs.get(sess, 0):.0f}s"]
        if meta.exists():
            kv = dict(line.split("=", 1) for line in meta.read_text().split())
            parts += [f"{k[2:]} {float(v):.0f}s" for k, v in kv.items() if k.startswith("t_")]
        return ", ".join(parts)

    def finish(sess: str, result) -> None:
        _, regions, fd, err = result[:4]
        shutil.rmtree(scratch / sess, ignore_errors=True)  # the whole point: never keep raw data
        if err and err.startswith("excluded"):
            with open(excluded_file, "a") as f:
                f.write(sess + "\n")
        stats["done" if not err else "problems"] += 1
        finished.append(time.time())
        n = stats["done"] + stats["problems"]
        # ETA from the recent pace (the first session also pays start-up costs)
        recent = finished[-11:]
        pace = (recent[-1] - recent[0]) / (len(recent) - 1) if len(recent) > 1 else recent[0] - t0
        eta_h = pace * (len(todo) - n) / 3600
        print(f"[{n}/{len(todo)}] {sess}: " + (err or f"{regions} regions, mean FD {fd:.2f} mm")
              + f"  [{stage_times(sess)}]  (eta {eta_h:.1f} h)", flush=True)

    stats, t0, finished, download_secs = {"done": 0, "problems": 0}, time.time(), [], {}
    threads = max(1, CORES // max(1, args.workers))
    print(f"{args.workers} workers x {threads} threads ({CORES} cores)")
    pool = (ProcessPoolExecutor(max_workers=args.workers, initializer=_limit_threads, initargs=(threads,))
            if args.workers > 0 else None)
    pending: dict[Future, str] = {}
    try:
        for s in todo:
            sess = s["experiment_id"]
            try:
                bold = fetch(s)
            except AccessDenied as exc:
                sys.exit(str(exc))
            except Exception as exc:
                print(f"download failed {sess}: {exc}")
                stats["problems"] += 1
                shutil.rmtree(scratch / sess, ignore_errors=True)
                continue
            if bold is None:
                print(f"skip {sess}: no resting-state BOLD")
                continue
            job = (sess, bold, out_dir, args.atlas, None)
            if pool is None:
                finish(sess, _one(job))
                continue
            pending[pool.submit(_one, job)] = sess
            # Hold at most `workers` raw downloads: wait before fetching more.
            while len(pending) >= args.workers:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    finish(pending.pop(f), f.result())
        for f in list(pending):
            finish(pending.pop(f), f.result())
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
        shutil.rmtree(scratch, ignore_errors=True)

    write_manifest(out_dir, labels)
    print(f"{stats['done']} processed, {stats['problems']} problems this run; raw data kept: none")
    if stats["problems"]:
        print("re-run the same command to retry the problems (finished sessions are skipped)")


if __name__ == "__main__":
    main()
