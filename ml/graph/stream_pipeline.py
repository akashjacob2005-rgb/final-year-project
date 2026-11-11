"""Download -> preprocess -> delete, one session at a time (low-disk mode).

The raw rs-fMRI for ~900 OASIS-3 sessions is 50-90 GB, but each session ends
up as one N x N connectivity matrix of a few KB. This script never keeps the
raw data: it downloads a session's BOLD into a scratch folder, runs the same
preprocessing as ml/graph/preprocess.py (motion correction, MNI registration,
atlas FC), saves processed_graph/<session>/fc.npy, and deletes the scratch
folder. Up to --downloads sessions download at once (NITRC is slow per
connection) while --workers sessions are processed; peak disk use is about
(downloads + workers) raw sessions, i.e. 1-2 GB.

    export OASIS_USER=... OASIS_PASS=...
    D=~/Downloads/7th_sem/PP-2/data
    python ml/graph/stream_pipeline.py --data-root $D --one-per-subject --max-normal 500 --limit 3
    caffeinate -i python ml/graph/stream_pipeline.py --data-root $D --one-per-subject --max-normal 500

Resumable: sessions whose fc.npy exists are skipped, so stop it any time and
re-run the same command. Sessions excluded (head motion, too few volumes) are
recorded with the reason in processed_graph/excluded.txt and not retried
unless --retry-excluded. Runs BOLD only (DTI needs
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
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
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
    ap.add_argument("--downloads", type=int, default=3,
                    help="sessions downloaded at the same time (NITRC is slow per connection; "
                         "1 = one after another)")
    ap.add_argument("--all-runs", action="store_true", help="download every resting-state run")
    ap.add_argument("--one-per-subject", action="store_true")
    ap.add_argument("--max-normal", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="stop after N new sessions")
    ap.add_argument("--retry-excluded", action="store_true",
                    help="redo sessions excluded by an earlier run (the old list is kept as a .bak)")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    out_dir = root / "processed_graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = root / "_stream_scratch"
    shutil.rmtree(scratch, ignore_errors=True)  # leftovers from an interrupted run
    scratch.mkdir()
    excluded_file = out_dir / "excluded.txt"
    # One "session<TAB>reason" line per exclusion (older files: session only).
    excluded_lines = excluded_file.read_text().splitlines() if excluded_file.exists() else []
    if args.retry_excluded and excluded_lines:
        excluded_file.rename(out_dir / f"excluded.{int(time.time())}.bak")
        print(f"--retry-excluded: {len(excluded_lines)} earlier exclusions moved aside and will be redone")
        excluded_lines = []
    excluded = {line.split()[0] for line in excluded_lines if line.strip()}
    if excluded_lines:
        reasons: dict[str, int] = {}
        for line in excluded_lines:
            parts = line.split("\t", 1)
            key = parts[1].split(":")[0] if len(parts) > 1 else "reason not recorded (older run)"
            key = "head motion" if "FD" in key or "FD" in (parts[1] if len(parts) > 1 else "") else key
            reasons[key] = reasons.get(key, 0) + 1
        print(f"{len(excluded)} sessions excluded earlier: "
              + ", ".join(f"{v} {k}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))

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
                f.write(f"{sess}\t{err}\n")
        stats["done" if not err else "problems"] += 1
        finished.append(time.time())
        if len(finished) % 25 == 0:
            # Checkpoint: a crash or reboot must not leave manifest.csv stale.
            write_manifest(out_dir, labels)
        n = stats["done"] + stats["problems"]
        # ETA from the recent pace (the first session also pays start-up costs)
        recent = finished[-11:]
        pace = (recent[-1] - recent[0]) / (len(recent) - 1) if len(recent) > 1 else recent[0] - t0
        eta_h = pace * (len(todo) - n) / 3600
        per_hour = 3600 / pace if pace > 0 else 0
        print(f"[{n}/{len(todo)}] {sess}: " + (err or f"{regions} regions, mean FD {fd:.2f} mm")
              + f"  [{stage_times(sess)}]  ({per_hour:.0f}/h, eta {eta_h:.1f} h)", flush=True)

    stats, t0, finished, download_secs = {"done": 0, "problems": 0}, time.time(), [], {}
    threads = max(1, CORES // max(1, args.workers))
    n_dl = max(1, args.downloads)
    max_raw = n_dl + max(1, args.workers)  # raw sessions allowed on disk at once
    print(f"{n_dl} parallel downloads, {args.workers} workers x {threads} threads ({CORES} cores); "
          f"at most {max_raw} raw sessions on disk")
    pool = (ProcessPoolExecutor(max_workers=args.workers, initializer=_limit_threads, initargs=(threads,))
            if args.workers > 0 else None)
    dl_pool = ThreadPoolExecutor(max_workers=n_dl)
    queue = list(todo)
    downloads: dict[Future, str] = {}
    processing: dict[Future, str] = {}
    raw_on_disk = 0
    try:
        while queue or downloads or processing:
            # Start downloads while there is room on disk and in the download pool.
            while queue and len(downloads) < n_dl and raw_on_disk < max_raw:
                s = queue.pop(0)
                downloads[dl_pool.submit(fetch, s)] = s["experiment_id"]
                raw_on_disk += 1
            done, _ = wait(list(downloads) + list(processing), return_when=FIRST_COMPLETED)
            for f in done:
                if f in downloads:
                    sess = downloads.pop(f)
                    try:
                        bold = f.result()
                    except AccessDenied as exc:
                        sys.exit(str(exc))
                    except Exception as exc:
                        print(f"download failed {sess}: {exc}")
                        stats["problems"] += 1
                        shutil.rmtree(scratch / sess, ignore_errors=True)
                        raw_on_disk -= 1
                        continue
                    if bold is None:
                        print(f"skip {sess}: no resting-state BOLD")
                        shutil.rmtree(scratch / sess, ignore_errors=True)
                        raw_on_disk -= 1
                        continue
                    job = (sess, bold, out_dir, args.atlas, None)
                    if pool is None:
                        finish(sess, _one(job))
                        raw_on_disk -= 1
                    else:
                        processing[pool.submit(_one, job)] = sess
                else:
                    finish(processing.pop(f), f.result())
                    raw_on_disk -= 1
    finally:
        dl_pool.shutdown(wait=False, cancel_futures=True)
        if pool:
            pool.shutdown(cancel_futures=True)
        shutil.rmtree(scratch, ignore_errors=True)

    write_manifest(out_dir, labels)
    print(f"{stats['done']} processed, {stats['problems']} problems this run; raw data kept: none")
    if stats["problems"]:
        print("re-run the same command to retry the problems (finished sessions are skipped)")


if __name__ == "__main__":
    main()
