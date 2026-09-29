"""Count what the Pitt download actually contains, before training on it.

    python ml/census_pitt.py

Read-only. Touches nothing, downloads nothing, writes one JSON summary.

WHY THIS EXISTS
Two assumptions baked into the pipeline need checking against the real corpus
rather than against folklore:

1. "~550 cookie recordings." The transcript bundle lists more sessions than have
   released audio, so the media count comes out lower. Training rows come from
   audio, not from transcripts, so the audio-backed count is the real sample
   size -- and it decides whether transcript-only sessions are worth pulling in
   to enlarge the text model.

2. "Cookie is balanced between the groups." If one group contributed far more
   recordings, that imbalance is itself a signal a classifier can lean on, and
   accuracy stops being a meaningful headline number.

It also reports the number that actually governs the error bars: DISTINCT
PARTICIPANTS. Pitt is longitudinal -- 002-0, 002-1, 002-2 are one person at
yearly visits -- so sessions overstate the sample. Cross-validation groups by
participant, and each fold tests on (participants / n_splits) people.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from download_pitt import MEDIA_EXTS, MIN_AUDIO_BYTES

HERE = Path(__file__).parent
PITT = HERE / "data" / "pitt"
TRANSCRIPTS = PITT / "transcripts"
MEDIA = PITT / "media"
MANIFEST = PITT / "download_manifest.csv"
OUT_JSON = PITT / "census.json"

GROUPS = ("control", "dementia")
TASKS = ("cookie", "fluency", "recall", "sentence")


def _classify(path: Path) -> tuple[str, str, str] | None:
    """-> (group, task, session stem), or None if the path is not a session."""
    parts = [p.lower() for p in path.parts]
    group = next((g for g in GROUPS if g in parts), None)
    task = next((t for t in TASKS if t in parts), None)
    if group is None or task is None:
        return None
    return group, task, path.stem


def _participant(stem: str) -> str:
    """'002-1' -> '002'; 'control/002-1' -> 'control/002'.

    Bare form matches build_dataset._meta's grouping key. The qualified form is
    used when pooling the groups, so that an ID reused across Control and
    Dementia is not silently counted as one person.
    """
    group, _, bare = stem.rpartition("/")
    pid = bare.partition("-")[0]
    return f"{group}/{pid}" if group else pid


def _transcript_index() -> dict[tuple[str, str], set[str]]:
    """(group, task) -> {session stems} from the extracted .cha tree."""
    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for p in TRANSCRIPTS.rglob("*.cha"):
        hit = _classify(p)
        if hit:
            group, task, stem = hit
            index[(group, task)].add(stem)
    return index


def _audio_index() -> dict[tuple[str, str], set[str]]:
    """(group, task) -> {session stems} that have a usable media file.

    Pitt ships WAV; MEDIA_EXTS is checked so a corpus with MP3 still counts. The
    size floor rejects the 11-byte stubs that media.talkbank.org returns for a
    GET with no Range header — see download_pitt.FULL_RANGE. Counting those as
    audio would report a complete corpus that cannot be transcribed.
    """
    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    media = (p for ext in MEDIA_EXTS for p in MEDIA.rglob(f"*{ext}"))
    for p in media:
        if p.stat().st_size < MIN_AUDIO_BYTES:
            continue
        hit = _classify(p)
        if hit:
            group, task, stem = hit
            index[(group, task)].add(stem)
    return index


def _manifest_tally() -> Counter:
    if not MANIFEST.exists():
        return Counter()
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        return Counter(row["status"] for row in csv.DictReader(fh))


def _participant_stats(stems: set[str]) -> dict:
    visits = Counter(_participant(s) for s in stems)
    if not visits:
        return {"participants": 0, "sessions": 0, "visits_per_participant": 0.0,
                "max_visits": 0, "repeat_attenders": 0}
    return {
        "participants": len(visits),
        "sessions": len(stems),
        "visits_per_participant": round(len(stems) / len(visits), 2),
        "max_visits": max(visits.values()),
        "repeat_attenders": sum(1 for v in visits.values() if v > 1),
    }


def main() -> int:
    if not TRANSCRIPTS.exists() and not MEDIA.exists():
        raise SystemExit(
            f"Nothing to count under {PITT}.\nRun: python ml/download_pitt.py"
        )

    tx = _transcript_index()
    au = _audio_index()

    print("=" * 66)
    print("SESSIONS PER TASK PER GROUP  (transcripts / with audio)")
    print("=" * 66)
    print(f"{'task':12}{'control':>18}{'dementia':>18}{'total':>16}")
    for task in TASKS:
        cells = []
        for group in GROUPS:
            n_tx = len(tx.get((group, task), ()))
            n_au = len(au.get((group, task), ()))
            cells.append(f"{n_tx:>10} / {n_au:<5}" if n_tx or n_au else f"{'-':>16} ")
        tot_tx = sum(len(tx.get((g, task), ())) for g in GROUPS)
        tot_au = sum(len(au.get((g, task), ())) for g in GROUPS)
        print(f"{task:12}{cells[0]:>18}{cells[1]:>18}{tot_tx:>9} / {tot_au:<5}")

    # ---- the cookie numbers that actually drive the training decision ----
    cookie_tx = {g: tx.get((g, "cookie"), set()) for g in GROUPS}
    cookie_au = {g: au.get((g, "cookie"), set()) for g in GROUPS}
    # Qualify by group before pooling: bare stems collide if the corpus reuses a
    # number across Control and Dementia, which would undercount sessions here
    # and, worse, merge two people into one CV group in build_dataset.py.
    all_tx = {f"{g}/{s}" for g in GROUPS for s in cookie_tx[g]}
    all_au = {f"{g}/{s}" for g in GROUPS for s in cookie_au[g]}
    transcript_only = all_tx - all_au

    print("\n" + "=" * 66)
    print("COOKIE THEFT — the task we train on")
    print("=" * 66)

    report: dict = {"cookie": {}, "per_task": {}}
    for group in GROUPS:
        stats = _participant_stats(cookie_au[group])
        report["cookie"][group] = {
            **stats,
            "transcripts": len(cookie_tx[group]),
            "transcript_only": len(cookie_tx[group] - cookie_au[group]),
        }
        print(
            f"  {group:9} {stats['sessions']:>4} audio sessions from "
            f"{stats['participants']:>4} participants "
            f"({stats['visits_per_participant']} visits each, "
            f"{stats['repeat_attenders']} returned)"
        )

    combined = _participant_stats(all_au)
    n_ctrl = report["cookie"]["control"]["sessions"]
    n_dem = report["cookie"]["dementia"]["sessions"]
    total = max(1, n_ctrl + n_dem)
    skew = abs(n_ctrl - n_dem) / total

    print(
        f"\n  TOTAL     {combined['sessions']:>4} audio sessions from "
        f"{combined['participants']:>4} participants"
    )
    print(f"  transcript-only (no audio released): {len(transcript_only)}")
    print(f"  class split: {n_ctrl} control / {n_dem} dementia "
          f"({100 * n_ctrl / total:.0f}% / {100 * n_dem / total:.0f}%)")

    # ---- what these numbers mean for the training plan -------------------
    print("\n" + "-" * 66)
    n_part = combined["participants"]
    per_fold = n_part / 5 if n_part else 0
    print(f"  Effective sample size is {n_part} PEOPLE, not "
          f"{combined['sessions']} recordings.")
    print(f"  5-fold grouped CV tests on ~{per_fold:.0f} people per fold — expect "
          "wide error bars.")
    if skew > 0.15:
        print(f"  ! Class skew is {100 * skew:.0f}%. Accuracy will flatter the "
              "majority class; read AUC and per-class recall instead.")
    else:
        print("  Classes are close to balanced, so accuracy is meaningful "
              "alongside AUC.")
    if len(transcript_only) > 0.25 * len(all_au):
        print(f"  ! {len(transcript_only)} sessions have a transcript but no audio "
              f"(+{100 * len(transcript_only) / max(1, len(all_au)):.0f}%).")
        print("    Worth deciding whether to train the text model on those too —")
        print("    they carry no pause markers, so they partly reintroduce the")
        print("    train/serve mismatch this pipeline exists to remove.")
    else:
        print("  Few transcript-only sessions; audio-only training loses little.")
    print("-" * 66)

    # ---- does an ID mean the same person in both groups? -----------------
    # build_dataset._meta groups by the bare digits, so this decides whether
    # that key is safe. The two possible causes need OPPOSITE fixes, which is
    # why this is measured rather than assumed:
    #   * separate numbering spaces -> two different people share a key, and
    #     StratifiedGroupKFold would wrongly tie them together. Qualify the key
    #     with the group.
    #   * a participant who converted from control to dementia between visits
    #     -> one real person, and qualifying the key would SPLIT them across
    #     train and test, reintroducing exactly the leak we are removing.
    ctrl_ids = {_participant(s) for s in cookie_au["control"]}
    dem_ids = {_participant(s) for s in cookie_au["dementia"]}
    shared_ids = sorted(ctrl_ids & dem_ids)
    if shared_ids:
        print(f"\n  ! {len(shared_ids)} participant ID(s) appear in BOTH groups: "
              f"{', '.join(shared_ids[:8])}{' ...' if len(shared_ids) > 8 else ''}")
        print("    build_dataset.py groups on the bare digits. Check the .cha")
        print("    headers: if age increments with the visit number and sex is")
        print("    constant it is one person who converted, and the bare key is")
        print("    correct. If they disagree it is separate numbering, and the key")
        print("    must be qualified by group.")
        if shared_ids == ["172"]:
            print("    -> 172 is the known Pitt case and is RESOLVED: one female")
            print("       followed at 72/73/74/75 who converted Control -> MCI.")
            print("       Keep the bare key. See build_dataset.py's docstring.")
    else:
        print("\n  No participant ID occurs in both groups — build_dataset.py's "
              "bare-digit grouping key is safe.")
    report["cookie"]["ids_in_both_groups"] = shared_ids

    tally = _manifest_tally()
    if tally:
        print(f"\n  download manifest: {dict(tally)}")

    for task in TASKS:
        report["per_task"][task] = {
            g: {"transcripts": len(tx.get((g, task), ())),
                "audio": len(au.get((g, task), ()))}
            for g in GROUPS
        }
    report["cookie"]["combined"] = combined
    report["cookie"]["transcript_only_sessions"] = len(transcript_only)
    report["cookie"]["class_skew"] = round(skew, 3)
    report["download_manifest"] = dict(tally)

    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
