"""ml/par_audio.py — interval algebra and WAV surgery.

Lives in the backend suite because that is the only pytest target in the repo,
even though par_audio is training-only code and never imported at inference. It
earns a test anyway: it decides what the model is trained on, and an off-by-one in
the interval arithmetic would quietly delete participant speech or leave
investigator speech in, neither of which raises.

Everything here is synthetic — generated WAVs and hand-written spans — so it runs
without the DementiaBank corpus, which cannot be committed.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from app.ml import paths  # noqa: F401  (side effect: puts ml/ on sys.path)
from par_audio import _merge, _subtract, drop_spans, spans  # noqa: E402

RATE = 8000


def _write_wav(path: Path, seconds: float, value: int = 1000) -> None:
    """A mono 16-bit WAV whose every sample is `value`, so cuts are measurable."""
    n = int(seconds * RATE)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(value.to_bytes(2, "little", signed=True) * n)


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


# --------------------------------------------------------------------------
# interval algebra
# --------------------------------------------------------------------------
def test_merge_collapses_overlapping_and_touching_ranges():
    assert _merge([(0, 5), (3, 8), (20, 25)]) == [(0, 8), (20, 25)]
    assert _merge([(10, 20), (0, 5)]) == [(0, 5), (10, 20)]
    # touching ranges merge: a zero-length gap is not a gap
    assert _merge([(0, 5), (5, 9)]) == [(0, 9)]
    assert _merge([]) == []


@pytest.mark.parametrize(
    "base,cut,expected",
    [
        ([(0, 10)], [(0, 10)], []),                      # identical
        ([(0, 10)], [(0, 3)], [(3, 10)]),                # head
        ([(0, 10)], [(7, 10)], [(0, 7)]),                # tail
        ([(0, 10)], [(4, 6)], [(0, 4), (6, 10)]),        # interior splits
        ([(0, 10)], [(20, 30)], [(0, 10)]),              # disjoint
        ([(0, 10)], [(2, 4), (6, 8)], [(0, 2), (4, 6), (8, 10)]),
        ([(0, 10)], [(-5, 5)], [(5, 10)]),               # cut starts before base
        ([(0, 10)], [(5, 50)], [(0, 5)]),                # cut ends after base
    ],
)
def test_subtract(base, cut, expected):
    assert _subtract(base, cut) == expected


def test_subtract_drops_zero_length_results():
    """A cut that exactly abuts must not leave an empty interval behind."""
    assert _subtract([(0, 10)], [(0, 5), (5, 10)]) == []


# --------------------------------------------------------------------------
# CHAT parsing
# --------------------------------------------------------------------------
def test_spans_reads_time_bullets_per_speaker(tmp_path):
    cha = tmp_path / "x.cha"
    cha.write_text(
        "@Begin\n"
        "@Participants:\tINV Investigator, PAR Participant\n"
        "*INV:\tthis picture . \x152410_3120\x15\n"
        "%mor:\tdet|this noun|picture .\n"
        "*PAR:\tokay . \x1512248_12747\x15\n"
        "@End\n"
    )
    out = spans(cha)
    assert out["INV"] == [(2410, 3120)]
    assert out["PAR"] == [(12248, 12747)]


def test_spans_carries_speaker_across_continuation_lines(tmp_path):
    """The bullet often lands on a wrapped line, not the '*PAR:' line itself."""
    cha = tmp_path / "x.cha"
    cha.write_text(
        "*PAR:\tthe little girl is laughing and the boy is\n"
        "\treaching for the cookie jar . \x151000_5000\x15\n"
        "%mor:\tdet|the adj|little noun|girl .\n"
    )
    out = spans(cha)
    assert out["PAR"] == [(1000, 5000)]


def test_dependent_tiers_contribute_no_spans(tmp_path):
    """%mor/%gra lines can contain digits; they must never register as audio."""
    cha = tmp_path / "x.cha"
    cha.write_text(
        "*PAR:\tokay . \x15100_200\x15\n"
        "%gra:\t1|2|DET 2|0|ROOT \x159999_99999\x15\n"
    )
    out = spans(cha)
    assert out["PAR"] == [(100, 200)]


def test_zero_length_bullets_are_ignored(tmp_path):
    cha = tmp_path / "x.cha"
    cha.write_text("*INV:\thm . \x15500_500\x15\n")
    assert spans(cha)["INV"] == []


# --------------------------------------------------------------------------
# WAV surgery
# --------------------------------------------------------------------------
def test_drop_spans_removes_exactly_the_requested_duration(tmp_path):
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    _write_wav(src, seconds=10.0)

    stats = drop_spans(src, dst, [(2000, 4000), (6000, 7000)])

    assert stats["total_ms"] == 10_000
    assert stats["removed_ms"] == 3000
    assert stats["kept_ms"] == 7000
    assert _duration(dst) == pytest.approx(7.0, abs=0.01)


def test_drop_spans_with_nothing_to_cut_is_a_faithful_copy(tmp_path):
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    _write_wav(src, seconds=3.0)
    stats = drop_spans(src, dst, [])
    assert stats["removed_ms"] == 0
    assert _duration(dst) == pytest.approx(3.0, abs=0.01)
    assert src.read_bytes()[44:] == dst.read_bytes()[44:]  # sample data identical


def test_drop_spans_preserves_wav_format(tmp_path):
    """Whisper reads this output; a changed rate would silently alter timing."""
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    _write_wav(src, seconds=2.0)
    drop_spans(src, dst, [(500, 900)])
    with wave.open(str(dst), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)


def test_drop_spans_clamps_spans_beyond_the_recording(tmp_path):
    """CHAT bullets can overrun the audio; that must not raise or truncate wrongly."""
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    _write_wav(src, seconds=2.0)
    stats = drop_spans(src, dst, [(1500, 99_000)])
    assert stats["kept_ms"] == pytest.approx(1500, abs=2)
    assert _duration(dst) == pytest.approx(1.5, abs=0.01)


def test_overlapping_inv_and_par_is_kept(tmp_path):
    """The rule that protects participant speech.

    INV time that coincides with PAR time is not cut, because overlapping speech
    cannot be separated and removing it would delete the participant's words.
    """
    inv = _merge([(1000, 5000)])
    par = _merge([(3000, 8000)])
    cut = _subtract(inv, par)
    assert cut == [(1000, 3000)]          # only the INV-only portion

    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    _write_wav(src, seconds=10.0)
    stats = drop_spans(src, dst, cut)
    assert stats["removed_ms"] == 2000    # not 4000
