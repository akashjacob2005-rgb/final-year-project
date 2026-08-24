"""Download the DementiaBank Pitt 'Cookie Theft' corpus (audio + transcripts).

This is the REAL corpus, not the scrubbed HuggingFace mirror that
ml/download_data.py fetches. The difference that matters: this one has audio, so
a model can learn how a person speaks and not merely what they said.

    python ml/download_pitt.py --limit 5     # smoke test
    python ml/download_pitt.py               # full corpus (~550 files)

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
files named .mp3. Authentication is a JSON POST that sets a cookie scoped to
Domain=talkbank.org, which then carries to media.talkbank.org.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
import zipfile
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

# Present in the body of every unauthenticated response.
AUTH_SHELL = b"initAuthModals"
ZIP_MAGIC = b"PK\x03\x04"

GROUPS = ("Control", "Dementia")
# Cookie Theft only, by default. Fluency, recall and sentence were administered
# to the dementia group ONLY, so a binary classifier trained on them would be
# reading task identity rather than speech — a perfect label leak.
DEFAULT_TASK = "cookie"

REQUEST_PAUSE_S = 0.25  # a small academic server, and a named account


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


def _sessions(cha_paths: list[Path], task: str) -> list[tuple[str, str, str]]:
    """-> (group, stem, url) for each session of the requested task."""
    out = []
    for p in cha_paths:
        parts = [x.lower() for x in p.parts]
        if task.lower() not in parts:
            continue
        group = next((g for g in GROUPS if g.lower() in parts), None)
        if group is None:
            continue
        out.append((group, p.stem, f"{MEDIA_ROOT}/{group}/{task}/{p.stem}.mp3"))
    return sorted(set(out))


def _download(s: requests.Session, url: str, dest: Path) -> tuple[str, int]:
    """Stream one file. Returns (status, bytes). Resumable and atomic."""
    if dest.exists() and dest.stat().st_size > 0:
        return "cached", dest.stat().st_size

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    r = s.get(url, timeout=120, stream=True)
    if r.status_code == 404:
        return "missing", 0

    first = next(r.iter_content(chunk_size=65536), b"")
    _guard(first, url)
    if first.lstrip()[:1] == b"<":
        return "not_audio", 0

    with part.open("wb") as fh:
        fh.write(first)
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)

    # Atomic rename: a killed download never leaves a truncated file that the
    # cache check above would later trust.
    part.replace(dest)
    return "ok", dest.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None, help="stop after N media files")
    ap.add_argument("--task", default=DEFAULT_TASK, help="cookie (default), fluency, recall, sentence")
    ap.add_argument("--group", choices=[*GROUPS, "both"], default="both")
    args = ap.parse_args()

    if args.task != DEFAULT_TASK:
        print(
            f"WARNING: '{args.task}' was administered to the dementia group only.\n"
            "         A binary model trained on it learns the task, not the speaker.\n"
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
    print(f"Media to fetch: {len(sessions)}  ({counts})\n")

    rows, tally = [], {}
    for i, (group, stem, url) in enumerate(sessions, 1):
        dest = MEDIA / group / args.task / f"{stem}.mp3"
        try:
            status, size = _download(s, url, dest)
        except requests.RequestException as exc:
            status, size = f"error:{type(exc).__name__}", 0

        tally[status] = tally.get(status, 0) + 1
        rows.append(
            {
                "group": group,
                "session": stem,
                "url": url,
                "path": str(dest.relative_to(OUT)),
                "bytes": size,
                "status": status,
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
                if status == "ok"
                else "",
            }
        )

        if i % 25 == 0 or i == len(sessions):
            print(f"  [{i}/{len(sessions)}] {tally}")
        if status not in ("cached",):
            time.sleep(REQUEST_PAUSE_S)

    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    got = tally.get("ok", 0) + tally.get("cached", 0)
    print(f"\nAudio available: {got}/{len(sessions)}")
    print(f"Manifest: {MANIFEST}")
    if got == 0:
        print("\nNothing downloaded — check the account has DementiaBank access.", file=sys.stderr)
        return 1
    print("\nNext: python ml/build_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
