"""Minimal client for the OASIS-3 project on XNAT (NITRC-IR).

OASIS-3 moved from XNAT Central (central.xnat.org, retired) to NITRC-IR:
https://www.nitrc.org/ir . It is still XNAT, so the REST paths are unchanged.
Access needs a NITRC account that has been granted the OASIS3 project
(apply via https://sites.wustl.edu/oasisbrains/). Credentials come only from
the environment and are never written to disk:

    export OASIS_USER=your_nitrc_login
    export OASIS_PASS=your_nitrc_password
    # optional, e.g. a mirror: export OASIS_XNAT_URL=https://www.nitrc.org/ir

Used by ml/mri/download_oasis.py (T1), ml/graph/download_oasis_func.py
(rs-fMRI / DWI) and ml/graph/build_session_list.py (which sessions exist).
"""

from __future__ import annotations

import csv
import io
import os
import sys
import time
import zipfile
from pathlib import Path

BASE = os.getenv("OASIS_XNAT_URL", "https://www.nitrc.org/ir").rstrip("/")
PROJECT = os.getenv("OASIS_PROJECT", "OASIS3")
RETRIES = 4


class AccessDenied(RuntimeError):
    """Login worked but the account cannot read this resource."""


class OasisXnat:
    def __init__(self, user: str | None = None, password: str | None = None, base: str = BASE,
                 session=None) -> None:
        import requests

        self.base = base.rstrip("/")
        self.s = session or requests.Session()
        user = user or os.getenv("OASIS_USER")
        password = password or os.getenv("OASIS_PASS")
        if not user or not password:
            sys.exit("Set OASIS_USER and OASIS_PASS (your NITRC login) environment variables first.")
        self.s.auth = (user, password)

    # -- plumbing -------------------------------------------------------
    def _get(self, path: str, timeout: int = 120, stream: bool = False):
        url = path if path.startswith("http") else f"{self.base}{path}"
        for attempt in range(RETRIES):
            try:
                r = self.s.get(url, timeout=timeout, stream=stream)
            except Exception:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(2 ** (attempt + 1))
                continue
            if r.status_code in (401, 403):
                raise AccessDenied(
                    f"HTTP {r.status_code} for {url} — this NITRC account cannot read it. "
                    f"Check that it has been granted the {PROJECT} project on NITRC-IR."
                )
            if r.status_code >= 500 and attempt < RETRIES - 1:
                time.sleep(2 ** (attempt + 1))  # NITRC maintenance windows are common
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"giving up on {url}")

    def _csv(self, path: str) -> list[dict]:
        sep = "&" if "?" in path else "?"
        return list(csv.DictReader(io.StringIO(self._get(f"{path}{sep}format=csv").text)))

    def login(self) -> None:
        r = self.s.post(f"{self.base}/data/JSESSION", timeout=30)
        if r.status_code != 200:
            sys.exit(f"XNAT login to {self.base} failed (HTTP {r.status_code}) — check the NITRC "
                     "username/password.")
        # Fail early, with a clear reason, if the project itself is not visible.
        self._get(f"/data/projects/{PROJECT}?format=json", timeout=60)

    # -- listings -------------------------------------------------------
    def mr_sessions(self) -> list[dict]:
        """All MR sessions of the project: [{label, subject}]."""
        rows = self._csv(f"/data/projects/{PROJECT}/experiments?xsiType=xnat:mrSessionData"
                         "&columns=label,subject_label")
        return [{"label": r.get("label", ""), "subject": r.get("subject_label", "")}
                for r in rows if r.get("label")]

    def scans(self, subject: str, session: str) -> list[dict]:
        return self._csv(f"/data/archive/projects/{PROJECT}/subjects/{subject}"
                         f"/experiments/{session}/scans")

    def subject_resources(self, subject: str) -> list[dict]:
        """Files attached to a subject (OASIS keeps project-wide CSVs on 0AS_data_files)."""
        out = []
        for exp in self._csv(f"/data/archive/projects/{PROJECT}/subjects/{subject}/experiments"):
            label = exp.get("label") or exp.get("ID")
            for f in self._csv(f"/data/archive/projects/{PROJECT}/subjects/{subject}"
                               f"/experiments/{label}/files"):
                out.append({"experiment": label, "name": f.get("Name", ""), "uri": f.get("URI", "")})
        return out

    def download(self, uri: str) -> bytes:
        return self._get(uri, timeout=600).content

    def download_scans(self, subject: str, session: str, scan_ids: list[str], out_root: Path,
                       keep_suffixes: tuple[str, ...]) -> int:
        """Download scans as one zip; extract only kept suffixes under out_root."""
        url = (f"/data/archive/projects/{PROJECT}/subjects/{subject}/experiments/{session}"
               f"/scans/{','.join(scan_ids)}/files?format=zip")
        data = self._get(url, timeout=1800).content
        n = 0
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for m in z.namelist():
                if not m.endswith(keep_suffixes):
                    continue
                target = out_root / m  # zip paths already start with the experiment label
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                n += 1
        return n
