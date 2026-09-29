"""Download the DementiaBank Pitt 'Cookie Theft' corpus (audio + transcripts).

This is the REAL corpus, not the scrubbed HuggingFace mirror that
ml/download_data.py fetches. The difference that matters: this one has audio, so
a model can learn how a person speaks and not merely what they said.

    python ml/download_pitt.py --limit 5     # smoke test
    python ml/download_pitt.py               # all cookie sessions

Not every session in the transcript bundle has released audio, so the media count
comes out below the transcript count. ml/census_pitt.py reports the split.

LICENCE — READ THIS
The Pitt corpus is distributed under the TalkBank data-use agreement. It must
not be committed to version control, redistributed, or baked into a container
image. ml/data/pitt/ is gitignored and dockerignored for exactly this reason.
Anyone reproducing this work needs their own DementiaBank approval:
https://talkbank.org/dementia/access/

Publications must cite Becker et al. (1994) and acknowledge NIH grants
AG03705 and AG05133.

WHY THE AUTH CODE LOOKS THE WAY IT DOES
TalkBank is not HTTP Basic. Every gated URL returns HTTP *200* with a ~319-byte
HTML login shell rather than a 401, so status codes are useless for detecting
failure — a naive downloader "succeeds" and writes hundreds of identical HTML
files named .wav. Authentication is a JSON POST that sets a cookie scoped to
Domain=talkbank.org, which then carries to media.talkbank.org.

THREE MORE WAYS THIS SILENTLY "SUCCEEDS"
Each of these produces a complete-looking corpus that cannot be trained on, so
each has an explicit check rather than a comment:

1. WRONG EXTENSION. Pitt audio is WAV. Every .mp3 URL 404s, which this script
   classifies as 'missing' — the same status as a session whose audio was never
   released. Guessing wrong therefore looks like "this corpus has no audio"
   rather than like a bug. See MEDIA_EXTS.
2. RANGELESS GET. media.talkbank.org answers a GET with no Range header with
   HTTP 206 and the first ELEVEN BYTES. That body starts with a valid RIFF
   header, so magic-byte sniffing passes and an 11-byte file gets written as
   'ok'. See FULL_RANGE.
3. UNVERIFIED LENGTH. Because (2) returns a success status with plausible
   content, the only reliable check is comparing bytes written against the total
   advertised in Content-Range. See _fetch's 'truncated' status.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    raise SystemExit("requests is required: pip install requests")

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

HERE = Path(__file__).parent
REPO = HERE.parent
OUT = HERE / "data" / "pitt"
TRANSCRIPTS = OUT / "transcripts"
MEDIA = OUT / "media"
MANIFEST = OUT / "download_manifest.csv"

LOGIN_URL = "https://sla2.talkbank.org/logInUser"
STATUS_URL = "https://sla2.talkbank.org/isLoggedIn"
ZIP_URL = "https://talkbank.org/data/dementia/English/Pitt?f=zip"
MEDIA_ROOT = "https://media.talkbank.org/dementia/English/Pitt"

# Pitt audio is released as WAV, not MP3 (~6.8MB per cookie session, 44.1kHz).
# Every .mp3 URL returns a real 404, so guessing the extension wrong shows up as
# 'missing' for all 500+ sessions rather than as an error. Tried in order; the
# first non-404 wins, so a corpus that does ship MP3 still works.
MEDIA_EXTS = (".wav", ".mp3")

# media.talkbank.org is an Express app that treats a GET with no Range header as
# a request for the first ELEVEN BYTES: it answers 206 with
# `Content-Range: bytes 0-10/6814562`. Without this header the downloader writes
# an 11-byte file, reports success, and the corpus looks complete while holding
# nothing but RIFF headers. Asking for bytes=0- returns the whole file.
FULL_RANGE = {"Range": "bytes=0-"}

# A real cookie recording is megabytes. Anything this small is a stub, not audio.
# Consumers (census_pitt, build_dataset) use this as the floor for "has audio".
MIN_AUDIO_BYTES = 4096

# Present in the body of every unauthenticated response.
AUTH_SHELL = b"initAuthModals"
ZIP_MAGIC = b"PK\x03\x04"

GROUPS = ("Control", "Dementia")
# Cookie Theft only, by default.
#
# Task availability, counted from the transcript bundle by ml/census_pitt.py
# rather than assumed:
#
#     task        control   dementia
#     cookie          243        309     <- the only task given to both groups
#     fluency           2        235
#     recall            1        263
#     sentence          1        236
#
# So cookie is not merely the preferred task, it is the only trainable one: the
# other three have one or two control transcripts each and cannot support a
# classifier at all. (An earlier comment here claimed all four tasks exist under
# both groups. Technically true, practically false, and the practical reading is
# the one that matters.) Two further reasons cookie is right regardless:
#
#   * The app administers exactly one task: describe the kitchen picture. The
#     whole point of this pipeline is that training text matches inference text,
#     and that argument extends to which task produced it.
#   * ml/features.py::CONTENT_UNITS is hard-coded to the Cookie Theft scene (boy,
#     girl, jar, stool, sink, curtain...). On a fluency word-list all four
#     content-unit features are identically zero for every subject, and
#     mean_sentence_length / subordinate_ratio / idea_density measure something
#     unrelated. Roughly a third of the feature vector would become noise.
DEFAULT_TASK = "cookie"

REQUEST_PAUSE_S = 0.25  # a small academic server, and a named account

# DO NOT RAISE THIS. Measured twice, the second time the hard way:
#
# media.talkbank.org caps this account at ~200-270 KB/s in AGGREGATE, not per
# connection. One connection sustains ~270 KB/s; four concurrent connections still
# total ~200 KB/s. A fresh connection briefly bursts near 900 KB/s for the first
# few MB, which is what makes it look parallelisable — it is not.
#
# Worse, sustained 4-way concurrency got this account cut off mid-run: after ~69
# files every subsequent request raised ConnectionError for the rest of the run.
# It lifted on its own within minutes, so it reads as throttling rather than a
# ban, but the parallelism bought nothing and cost the run. Two workers only
# overlap request latency.
DEFAULT_WORKERS = 2

# Retry policy. The throttle above is transient, so a failed session must be
# retried rather than recorded as permanently absent — the earlier run marked 483
# perfectly available sessions 'error:ConnectionError' purely because nothing
# retried them.
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 3.0  # 3s, 6s, 12s

# Circuit breaker. Once the server stops talking to us, every remaining session
# fails in milliseconds; the earlier run burned through 483 of them in minutes and
# reported them all as failures. Stopping instead leaves them simply not-yet-
# downloaded, so re-running resumes cleanly.
CONSECUTIVE_FAILURE_LIMIT = 10

_local = threading.local()


def _credentials() -> tuple[str, str]:
    if load_dotenv is not None:
        load_dotenv(REPO / ".env")
    email = os.getenv("TALKBANK_EMAIL", "").strip()
    password = os.getenv("TALKBANK_PASSWORD", "").strip()
    if not email or not password:
        raise SystemExit(
            "Missing TalkBank credentials.\n\n"
            "Add these to the .env file at the repository root:\n"
            "    TALKBANK_EMAIL=you@university.edu\n"
            "    TALKBANK_PASSWORD=your-password\n\n"
            "DementiaBank access is password-protected and must be approved "
            "first: https://talkbank.org/dementia/access/\n"
            "(.env is gitignored; credentials are never written to the manifest "
            "or logged.)"
        )
    return email, password


def login() -> requests.Session:
    """Authenticate and return a session carrying the talkbank cookie."""
    email, password = _credentials()
    s = requests.Session()
    s.headers["User-Agent"] = "NeuroGuard-research/1.0"

    r = s.post(LOGIN_URL, json={"email": email, "pswd": password}, timeout=30)
    try:
        body = r.json()
    except ValueError:
        raise SystemExit(f"Unexpected login response ({r.status_code}); not JSON.")
    if not body.get("success"):
        raise SystemExit(
            f"TalkBank login failed: {body.get('respMsg', 'unknown reason')}\n"
            "Check TALKBANK_EMAIL / TALKBANK_PASSWORD."
        )

    # Logging in is not the same as being authorised for DementiaBank. Separating
    # the two turns a baffling 'all downloads are HTML' failure into a clear one.
    try:
        st = s.post(STATUS_URL, json={}, timeout=30).json().get("authStatus", {})
        if st and not st.get("authorized", True):
            raise SystemExit(
                "Signed in, but this account is not authorised for DementiaBank.\n"
                "Membership is granted per-corpus: https://talkbank.org/dementia/access/"
            )
    except SystemExit:
        raise
    except Exception:
        pass  # advisory only

    print(f"Authenticated as {email}")
    return s


def _worker_session(master: requests.Session) -> requests.Session:
    """A per-thread Session sharing the authenticated cookie.

    requests.Session is not documented as thread-safe, so workers do not share
    one. Logging in N times would be the obvious alternative, but the cookie is
    the whole authentication state — copying it means one login and N independent
    connection pools.
    """
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(master.headers)
        s.cookies.update(master.cookies)
        _local.session = s
    return s


def _guard(content: bytes, url: str) -> None:
    """Reject the login shell that TalkBank serves with a 200 when unauthorised."""
    if AUTH_SHELL in content[:2048]:
        raise SystemExit(
            f"Got the TalkBank login page instead of data:\n  {url}\n"
            "The session expired or this account lacks DementiaBank access."
        )


def fetch_transcripts(s: requests.Session) -> list[Path]:
    """Download and extract the transcript bundle; return the .cha paths.

    The zip is fetched mainly for its file list: it is one request and it gives
    the authoritative inventory of sessions, from which media URLs are derived.
    Scraping the media directory listings instead would not work — they are
    rendered by JavaScript behind the same auth gate.
    """
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    archive = OUT / "Pitt.zip"

    if not archive.exists():
        print(f"Downloading transcripts: {ZIP_URL}")
        r = s.get(ZIP_URL, timeout=300)
        _guard(r.content, ZIP_URL)
        if not r.content.startswith(ZIP_MAGIC):
            raise SystemExit("Transcript download is not a zip archive.")
        archive.write_bytes(r.content)
        print(f"  {len(r.content):,} bytes")
    else:
        print(f"Using cached {archive.name}")

    with zipfile.ZipFile(archive) as z:
        z.extractall(TRANSCRIPTS)

    return sorted(TRANSCRIPTS.rglob("*.cha"))


def _sessions(cha_paths: list[Path], task: str) -> list[tuple[str, str]]:
    """-> (group, stem) for each session of the requested task.

    The URL is no longer built here: the extension is discovered per session by
    _download, which tries MEDIA_EXTS in order.
    """
    out = []
    for p in cha_paths:
        parts = [x.lower() for x in p.parts]
        if task.lower() not in parts:
            continue
        group = next((g for g in GROUPS if g.lower() in parts), None)
        if group is None:
            continue
        out.append((group, p.stem))
    return sorted(set(out))


def _expected_size(r: requests.Response) -> int | None:
    """Total file size from Content-Range ('bytes 0-6814561/6814562') or length."""
    rng = r.headers.get("Content-Range", "")
    if "/" in rng:
        total = rng.rsplit("/", 1)[1].strip()
        if total.isdigit():
            return int(total)
    cl = r.headers.get("Content-Length", "")
    return int(cl) if cl.isdigit() else None


def _fetch_once(s: requests.Session, url: str, dest: Path) -> tuple[str, int]:
    """One attempt. Returns (status, bytes). Atomic; verifies the full length."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    r = s.get(url, timeout=300, stream=True, headers=FULL_RANGE)
    if r.status_code == 404:
        return "missing", 0

    first = next(r.iter_content(chunk_size=65536), b"")
    _guard(first, url)
    if first.lstrip()[:1] == b"<":
        return "not_audio", 0

    expected = _expected_size(r)
    with part.open("wb") as fh:
        fh.write(first)
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)

    got = part.stat().st_size
    # The 11-byte default-range response is a 206 with valid audio magic, so
    # nothing above it can catch it. Comparing against the advertised total is
    # what separates a real download from a truncated one.
    if expected is not None and got < expected:
        part.unlink(missing_ok=True)
        return f"truncated:{got}/{expected}", 0

    # Atomic rename: a killed download never leaves a truncated file that the
    # cache check in _download would later trust.
    part.replace(dest)
    return "ok", got


def _fetch(s: requests.Session, url: str, dest: Path) -> tuple[str, int]:
    """Stream one URL to dest, retrying transient failures.

    'missing' (404) and 'not_audio' are answers, not failures, so they return
    immediately. Connection errors and truncated bodies are the throttle showing
    up, and those are worth waiting out: the alternative is recording an available
    session as permanently absent, which is exactly how an earlier run lost 483 of
    them.
    """
    delay = BACKOFF_BASE_S
    last = "error:unknown"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            status, size = _fetch_once(s, url, dest)
            if status == "ok" or status in ("missing", "not_audio"):
                return status, size
            last = status  # truncated -> retry
        except requests.RequestException as exc:
            last = f"error:{type(exc).__name__}"

        if attempt < MAX_ATTEMPTS:
            time.sleep(delay)
            delay *= 2
    return last, 0


def _download(
    s: requests.Session, group: str, task: str, stem: str, out_dir: Path
) -> tuple[str, int, str]:
    """Fetch one session, trying each candidate extension.

    Returns (status, bytes, url_used). Resumable: an already-complete file in any
    of the candidate extensions short-circuits the request entirely.
    """
    for ext in MEDIA_EXTS:
        dest = out_dir / f"{stem}{ext}"
        if dest.exists() and dest.stat().st_size > 0:
            return "cached", dest.stat().st_size, str(dest)

    last = "missing"
    for ext in MEDIA_EXTS:
        url = f"{MEDIA_ROOT}/{group}/{task}/{stem}{ext}"
        status, size = _fetch(s, url, out_dir / f"{stem}{ext}")
        if status == "ok":
            return status, size, url
        last = status
        if status != "missing":
            # A non-404 failure is about this file, not the extension guess.
            return status, 0, url
    return last, 0, f"{MEDIA_ROOT}/{group}/{task}/{stem}{MEDIA_EXTS[0]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None, help="stop after N media files")
    ap.add_argument("--task", default=DEFAULT_TASK, help="cookie (default), fluency, recall, sentence")
    ap.add_argument("--group", choices=[*GROUPS, "both"], default="both")
    ap.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"concurrent downloads (default {DEFAULT_WORKERS}); see DEFAULT_WORKERS",
    )
    args = ap.parse_args()

    if args.task != DEFAULT_TASK:
        print(
            f"WARNING: '{args.task}' is not the task the app administers, and the\n"
            "         Cookie Theft content-unit features do not apply to it.\n"
            "         See the DEFAULT_TASK comment before training on this.\n"
        )

    print(
        "DementiaBank Pitt corpus — TalkBank data-use agreement applies.\n"
        "Do not commit, redistribute, or containerise this data.\n"
        "Cite Becker et al. (1994); acknowledge NIH AG03705 and AG05133.\n"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    s = login()

    cha = fetch_transcripts(s)
    print(f"Transcripts: {len(cha)} .cha files")

    sessions = _sessions(cha, args.task)
    if args.group != "both":
        sessions = [x for x in sessions if x[0] == args.group]
    if args.limit:
        sessions = sessions[: args.limit]

    if not sessions:
        raise SystemExit(f"No '{args.task}' sessions found in the transcript bundle.")

    counts = {g: sum(1 for x in sessions if x[0] == g) for g in GROUPS}
    print(f"Media to fetch: {len(sessions)}  ({counts})")
    print(f"Workers: {args.workers}\n")

    stop = threading.Event()

    def job(item: tuple[str, str]) -> dict:
        group, stem = item
        out_dir = MEDIA / group / args.task
        if stop.is_set():
            # Not an error: simply not attempted, so a re-run picks it up.
            return {"group": group, "session": stem, "url": "", "path": "",
                    "bytes": 0, "status": "skipped", "sha256": ""}
        try:
            status, size, url = _download(
                _worker_session(s), group, args.task, stem, out_dir
            )
        except requests.RequestException as exc:
            status, size, url = f"error:{type(exc).__name__}", 0, ""

        dest = Path(url) if status == "cached" else out_dir / Path(url).name
        if status != "cached":
            time.sleep(REQUEST_PAUSE_S)
        return {
            "group": group,
            "session": stem,
            "url": url,
            "path": str(dest.relative_to(OUT)) if dest.is_relative_to(OUT) else "",
            "bytes": size,
            "status": status,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
            if status in ("ok", "cached") and dest.exists()
            else "",
        }

    rows, tally = [], {}
    consecutive = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(job, item) for item in sessions]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            tally[row["status"]] = tally.get(row["status"], 0) + 1

            # 'missing' is a real answer about this corpus, not a transport
            # failure, so it must not trip the breaker — plenty of Pitt sessions
            # genuinely have no released audio.
            if row["status"].startswith(("error:", "truncated")):
                consecutive += 1
                if consecutive >= CONSECUTIVE_FAILURE_LIMIT and not stop.is_set():
                    stop.set()
                    print(
                        f"\n  ! {consecutive} consecutive transport failures — "
                        "stopping.\n"
                        "    Every request is failing, so continuing would only "
                        "mark the rest\n"
                        "    unavailable. Already-downloaded files are kept; "
                        "re-run to resume."
                    )
            elif row["status"] != "skipped":
                consecutive = 0

            if i % 25 == 0 or i == len(sessions):
                rate = i / max(time.time() - t0, 1e-9)
                eta = (len(sessions) - i) / rate / 60
                print(f"  [{i}/{len(sessions)}] {tally}  ~{eta:.0f} min left")

    # Deterministic manifest regardless of completion order.
    rows.sort(key=lambda r: (r["group"], r["session"]))

    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    got = tally.get("ok", 0) + tally.get("cached", 0)
    unfinished = tally.get("skipped", 0) + sum(
        n for st, n in tally.items() if st.startswith(("error:", "truncated"))
    )

    print(f"\nAudio available: {got}/{len(sessions)}")
    print(f"Manifest: {MANIFEST}")
    if got == 0:
        print("\nNothing downloaded — check the account has DementiaBank access.", file=sys.stderr)
        return 1

    # An incomplete run must not look like a finished one: the next step trains a
    # model, and silently training on whatever arrived is how a corpus ends up
    # missing an entire class.
    if unfinished:
        print(
            f"\n! {unfinished} sessions did not complete (transport failures or "
            "skipped after the\n"
            "  breaker tripped). This run is INCOMPLETE — re-run this script to "
            "resume before\n"
            "  training; downloaded files are cached, so it only fetches what is "
            "still absent."
        )
        print("\nThen: python ml/census_pitt.py")
        return 1

    print("\nNext: python ml/census_pitt.py && python ml/build_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
