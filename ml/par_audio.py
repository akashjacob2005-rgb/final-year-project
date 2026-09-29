"""Remove the investigator's speech from each Pitt recording.

    python ml/par_audio.py            # media/ -> media_par/
    python ml/par_audio.py --report   # measure the confound, write nothing

WHY THIS EXISTS — A CLASS-CORRELATED CONFOUND
Pitt sessions are interviews. Whisper transcribes the whole file, so the
investigator's prompts land in the training text alongside the participant's
description. That would merely be noise if it were evenly spread. It is not:

    INV share of speaking time   Control median  4.1%   Dementia median 12.0%
    INV share of words           Control median  8.8%   Dementia median 16.4%

The examiner talks roughly three times more with dementia participants, because
they need more prompting. So a model fitted on the raw transcript can score the
condition by detecting HOW MUCH THE EXAMINER SPOKE — a leak that is correlated
with the label and therefore inflates the reported AUC in the flattering
direction.

It is also a train/serve mismatch of exactly the kind this pipeline exists to
remove: in the app the user records alone, so a speaker present in every training
example is absent from every inference example.

The timing features are affected worse than the text. pause_ratio over a raw
recording counts turn-taking silence — the gaps around the examiner's turns — as
participant hesitation. On one control session it read 0.475, most of which was
not the participant pausing at all.

WHAT IT DOES
CHAT files carry millisecond time bullets (\\x15start_end\\x15) per utterance, so
INV spans are known exactly. Those spans are cut out of the WAV and the remainder
is concatenated. Participant silence is preserved; only examiner speech is
removed, which is why the gap collapses where the examiner was talking — that gap
was not the participant thinking.

Spans where INV and PAR overlap are KEPT. Overlapping speech cannot be separated
by cutting, and discarding it would delete participant words.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import wave
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from download_pitt import MEDIA_EXTS, MIN_AUDIO_BYTES  # noqa: E402

PITT = HERE / "data" / "pitt"
TRANSCRIPTS = PITT / "transcripts"
MEDIA = PITT / "media"
MEDIA_PAR = PITT / "media_par"
REPORT = PITT / "par_audio_report.json"

GROUPS = ("Control", "Dementia")
TASK = "cookie"

# CHAT time bullet: NAK, start_end in ms, NAK.
BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")


def spans(cha: Path) -> dict[str, list[tuple[int, int]]]:
    """-> {"PAR": [(start_ms, end_ms), ...], "INV": [...]}.

    Utterances can wrap onto continuation lines (leading tab) with the bullet on
    a later line, so the speaker is carried forward until the next '*' or '%'
    line rather than read per line.
    """
    out: dict[str, list[tuple[int, int]]] = {"PAR": [], "INV": []}
    speaker = None
    for line in cha.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("*"):
            speaker = line[1:4]
        elif line.startswith(("%", "@")):
            speaker = None          # dependent tiers carry no audio of their own
        if speaker in out:
            for m in BULLET.finditer(line):
                a, b = int(m.group(1)), int(m.group(2))
                if b > a:
                    out[speaker].append((a, b))
    return out


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [list(ranges[0])]
    for a, b in ranges[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _subtract(base: list[tuple[int, int]], cut: list[tuple[int, int]]):
    """base minus cut, both merged and sorted."""
    out = []
    for a, b in base:
        pieces = [(a, b)]
        for ca, cb in cut:
            nxt = []
            for pa, pb in pieces:
                if cb <= pa or ca >= pb:      # disjoint
                    nxt.append((pa, pb))
                    continue
                if pa < ca:
                    nxt.append((pa, ca))
                if cb < pb:
                    nxt.append((cb, pb))
            pieces = nxt
        out.extend(pieces)
    return [(a, b) for a, b in out if b > a]


def _read_pcm(path: Path) -> tuple[bytes, int, int, int]:
    """-> (frames, channels, sample_width, framerate) for any audio container.

    Pitt is almost entirely WAV, which stdlib `wave` reads with no decode. Exactly
    one session (Control/304-1) is served as MP3 — the MEDIA_EXTS fallback in
    download_pitt.py catches it — so a decoder is needed for completeness. PyAV is
    already present as faster-whisper's decoder, which keeps this dependency-free.
    """
    try:
        with wave.open(str(path), "rb") as w:
            return (
                w.readframes(w.getnframes()),
                w.getnchannels(), w.getsampwidth(), w.getframerate(),
            )
    except (wave.Error, EOFError):
        pass

    import av  # noqa: PLC0415 — only needed for the rare non-WAV session

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(bytes(out.planes[0]))
        # Flush the resampler, or the final partial frame is lost.
        for out in resampler.resample(None):
            chunks.append(bytes(out.planes[0]))
    return b"".join(chunks), 1, 2, 16000


def drop_spans(wav_in: Path, wav_out: Path, drop_ms: list[tuple[int, int]]) -> dict:
    """Write wav_in minus drop_ms to wav_out. Returns byte/ms accounting."""
    frames, nch, sw, fr = _read_pcm(wav_in)
    nframes = len(frames) // (nch * sw)

    bytes_per_frame = nch * sw
    total_ms = int(1000 * nframes / fr)

    def to_frame(ms: int) -> int:
        return max(0, min(nframes, int(round(ms * fr / 1000.0))))

    keep = _subtract([(0, total_ms)], _merge(drop_ms))
    chunks = []
    kept_ms = 0
    for a, b in keep:
        fa, fb = to_frame(a), to_frame(b)
        if fb > fa:
            chunks.append(frames[fa * bytes_per_frame: fb * bytes_per_frame])
            kept_ms += b - a

    wav_out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_out), "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(sw)
        w.setframerate(fr)
        w.writeframes(b"".join(chunks))

    return {
        "total_ms": total_ms,
        "kept_ms": kept_ms,
        "removed_ms": total_ms - kept_ms,
        "framerate": fr,
        "channels": nch,
    }


def _audio_for(group: str, stem: str) -> Path | None:
    for ext in MEDIA_EXTS:
        p = MEDIA / group / TASK / f"{stem}{ext}"
        if p.exists() and p.stat().st_size >= MIN_AUDIO_BYTES:
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true",
                    help="measure the INV share only; write no audio")
    ap.add_argument("--force", action="store_true",
                    help="rewrite outputs that already exist")
    args = ap.parse_args()

    rows, skipped = [], []
    for group in GROUPS:
        cha_dir = TRANSCRIPTS / group / TASK
        if not cha_dir.is_dir():
            continue
        for cha in sorted(cha_dir.glob("*.cha")):
            stem = cha.stem
            src = _audio_for(group, stem)
            if src is None:
                continue                      # no audio released, or not downloaded

            sp = spans(cha)
            inv, par = _merge(sp["INV"]), _merge(sp["PAR"])
            if not par and not inv:
                # No time alignment at all: nothing can be cut safely, so the file
                # passes through unchanged and is counted so coverage is honest.
                skipped.append(f"{group}/{stem}")
                if not args.report:
                    dst = MEDIA_PAR / group / TASK / f"{stem}.wav"
                    if args.force or not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(src.read_bytes())
                continue

            # Overlapping speech cannot be separated by cutting, so INV time that
            # coincides with PAR time is kept rather than deleted with it.
            cut = _subtract(inv, par)

            rec = {"group": group, "session": stem,
                   "inv_ms": sum(b - a for a, b in inv),
                   "par_ms": sum(b - a for a, b in par),
                   "cut_ms": sum(b - a for a, b in cut)}

            if not args.report:
                dst = MEDIA_PAR / group / TASK / f"{stem}.wav"
                if args.force or not dst.exists():
                    try:
                        rec.update(drop_spans(src, dst, cut))
                    except wave.Error as exc:
                        rec["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(rec)

    if not rows:
        raise SystemExit("No aligned sessions with audio found. Run ml/download_pitt.py")

    print(f"{'group':10}{'n':>6}{'INV share of speech':>22}{'cut':>10}")
    for group in GROUPS:
        g = [r for r in rows if r["group"] == group]
        if not g:
            continue
        share = [r["inv_ms"] / max(r["inv_ms"] + r["par_ms"], 1) for r in g]
        cut = sum(r["cut_ms"] for r in g) / 1000 / 60
        print(f"{group:10}{len(g):>6}{statistics.median(share):>21.1%}{cut:>9.1f}m")

    errors = [r for r in rows if "error" in r]
    print(f"\nsessions processed : {len(rows)}")
    print(f"unaligned (copied) : {len(skipped)}")
    print(f"errors             : {len(errors)}")
    for r in errors[:5]:
        print(f"  ! {r['group']}/{r['session']}: {r['error']}")

    REPORT.write_text(json.dumps(
        {"sessions": rows, "unaligned": skipped}, indent=2))
    print(f"\nReport -> {REPORT}")
    if not args.report:
        print(f"Audio  -> {MEDIA_PAR}")
        print("\nNext: python ml/build_dataset.py --force   (re-transcribe PAR-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
