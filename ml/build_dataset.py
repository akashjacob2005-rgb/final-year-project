"""Transcribe the Pitt corpus with the SAME code path the app uses at inference.

    python ml/build_dataset.py              # all downloaded audio
    python ml/build_dataset.py --workers 6  # tune parallelism
    python ml/build_dataset.py --force      # ignore the cache and redo

Why this matters more than it looks: the deployed model is currently trained on
clinician-typed CHAT transcripts but *evaluated in production* on Whisper output.
That mismatch is the top caveat in ml/artifacts/metrics.json and the stated
reason the weaker `combined+logreg` was deployed over `tfidf+logreg`. Running the
training corpus through the identical ASR path removes the mismatch rather than
apologising for it.

"Identical" is literal. This module imports backend/app/ml/speech.py and calls
its functions — same faster-whisper model, same beam size, same VAD, same
normalise_transcript, same _timing_features. Not a copy: the same objects. The
backend file is not modified, and does not need to be; it has no relative
imports, so it is importable on its own.

Labels and grouping come from the filesystem, so no CHAT parsing is required:
    media/Control/cookie/002-1.mp3  ->  label 0, participant "002", visit 1
    media/Dementia/cookie/017-0.mp3 ->  label 1, participant "017", visit 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent
BACKEND_ML = REPO / "backend" / "app" / "ml"

PITT = HERE / "data" / "pitt"
MEDIA = PITT / "media"
ASR_CACHE = PITT / "asr"
DATASET = PITT / "dataset.csv"

LABEL_OF_GROUP = {"control": 0, "dementia": 1}


def _speech():
    """Import the backend's speech module, unmodified.

    speech.py has no relative imports, so putting its directory on sys.path is
    enough — there is no need to move it into ml/ or duplicate it here, and the
    running app is completely untouched by this script.
    """
    if str(BACKEND_ML) not in sys.path:
        sys.path.insert(0, str(BACKEND_ML))
    import speech  # noqa: PLC0415

    return speech


def _meta(path: Path) -> dict | None:
    """Derive label / participant / visit from the file's location and name."""
    parts = [p.lower() for p in path.parts]
    group = next((g for g in LABEL_OF_GROUP if g in parts), None)
    if group is None:
        return None

    stem = path.stem                      # "002-1"
    participant, _, visit = stem.partition("-")
    if not participant.isdigit():
        return None

    return {
        "session_id": f"{group}/{stem}",
        "participant_id": participant,     # <- the CV grouping key
        "visit": int(visit) if visit.isdigit() else 0,
        "group": group,
        "label": LABEL_OF_GROUP[group],
    }


def _transcribe_one(args: tuple[str, str]) -> dict:
    """Worker body. Runs in its own process with its own WhisperModel.

    speech.py serialises Whisper behind a threading.Lock because the model is not
    thread-safe. Processes sidestep that entirely — each has its own interpreter,
    its own lock, and its own model — so the corpus transcribes in parallel
    without changing a line of the backend.
    """
    path_str, cache_str = args
    cache = Path(cache_str)
    if cache.exists():
        return {"status": "cached", "path": path_str}

    try:
        speech = _speech()
        out = speech.transcribe(path_str)
        out["whisper_model"] = os.getenv("WHISPER_MODEL", "base.en")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
        return {"status": "ok", "path": path_str}
    except Exception as exc:  # noqa: BLE001 — one bad file must not stop the run
        return {"status": "error", "path": path_str, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="ignore cached transcripts")
    args = ap.parse_args()

    if not MEDIA.exists():
        raise SystemExit("No audio found. Run: python ml/download_pitt.py")

    audio = sorted(p for p in MEDIA.rglob("*.mp3") if p.stat().st_size > 0)
    rows = [(p, _meta(p)) for p in audio]
    rows = [(p, m) for p, m in rows if m][: args.limit]
    if not rows:
        raise SystemExit(f"No usable audio under {MEDIA}")

    if args.force:
        for f in ASR_CACHE.rglob("*.json"):
            f.unlink()

    ASR_CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(p), str(ASR_CACHE / f"{m['session_id'].replace('/', '_')}.json"))
        for p, m in rows
    ]

    print(f"Transcribing {len(jobs)} recordings with {args.workers} workers")
    print(f"  model: {os.getenv('WHISPER_MODEL', 'base.en')} (the app's default)\n")

    tally: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_transcribe_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            tally[res["status"]] = tally.get(res["status"], 0) + 1
            if res["status"] == "error":
                print(f"  ! {Path(res['path']).name}: {res['error']}")
            if i % 25 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {tally}")

    # --- assemble the training table ------------------------------------
    speech = _speech()
    timing_keys = list(speech._timing_features([], 0.0).keys())

    out_rows = []
    for p, m in rows:
        cache = ASR_CACHE / f"{m['session_id'].replace('/', '_')}.json"
        if not cache.exists():
            continue
        data = json.loads(cache.read_text())
        timing = data.get("timing", {})
        out_rows.append(
            {
                **m,
                "transcript": data.get("transcript", ""),
                "raw_transcript": data.get("raw_transcript", ""),
                **{k: timing.get(k, 0.0) for k in timing_keys},
            }
        )

    if not out_rows:
        raise SystemExit("No transcripts produced.")

    with DATASET.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    n_dem = sum(r["label"] for r in out_rows)
    participants = {r["participant_id"] for r in out_rows}
    print(f"\nWrote {DATASET}")
    print(f"  sessions     : {len(out_rows)}  (dementia={n_dem} control={len(out_rows)-n_dem})")
    print(f"  participants : {len(participants)}")
    # The gap between these two numbers is the whole reason training must group
    # by participant: the same person recurs across yearly visits, so a random
    # split would put one visit in train and another in test.
    print(f"  visits/person: {len(out_rows)/max(len(participants),1):.2f}")
    print("\nNext: python ml/train_audio_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
