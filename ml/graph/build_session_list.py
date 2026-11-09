"""List every OASIS-3 MR session usable for the graph model, with a CDR label.

    export OASIS_USER=... OASIS_PASS=...          # NITRC login with OASIS3 access
    python ml/graph/build_session_list.py --data-root ~/Downloads/7th_sem/PP-2/data

1. Asks NITRC-IR for all OASIS3 MR sessions and, per session, its scans
   (cached in processed/oasis3_scan_index.json, so a re-run only asks about
   new sessions). Keeps sessions with a resting-state BOLD run; notes DWI.
2. Finds OASIS-3's clinical CDR table among the project-wide files on the
   `0AS_data_files` subject (or takes --clinical-csv if you downloaded it
   yourself from the OASIS site) and labels each session with the CDR
   assessed closest in time, within --max-days (default 365):
       CDR 0 -> 0 Normal, 0.5 -> 1 MCI, >= 1 -> 2 Dementia.
   Sessions with no CDR that close are dropped (a label a year off is noise).

Writes processed/graph_sessions.csv (what to download, with scan ids) and
processed/graph_session_labels.csv (mr_session, label, cdr, days_apart), and
prints the class counts, which decide whether training is worth running.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/

DATA_FILES_SUBJECT = "0AS_data_files"
SUBJECT_COLS = ("OASISID", "OASIS_ID", "Subject", "subject_id", "subject")
DAYS_COLS = ("days_to_visit", "days", "Days", "days_since_entry")
CDR_COLS = ("CDRTOT", "cdr", "CDR", "CDRGLOB", "cdr_global", "CDR_GLOBAL")
LABEL_COLS = ("OASIS_session_label", "ADRC_ADRCCLINICALDATA ID", "label", "Label")
_DAYS_RE = re.compile(r"_d(\d+)$")


def is_rest_bold(scan_type: str) -> bool:
    t = (scan_type or "").lower()
    return "bold" in t and ("rest" in t or "task" not in t)


def is_dwi(scan_type: str) -> bool:
    t = (scan_type or "").lower()
    return "dwi" in t or "dti" in t


def session_days(label: str) -> int | None:
    """Days since entry encoded in OASIS labels, e.g. OAS30001_MR_d0129 -> 129."""
    m = _DAYS_RE.search(label or "")
    return int(m.group(1)) if m else None


def _pick(row: dict, names: tuple[str, ...]):
    for n in names:
        if n in row and str(row[n]).strip() != "":
            return row[n]
    return None


def parse_cdr_table(rows: list[dict]) -> dict[str, list[tuple[int, float]]]:
    """subject -> [(days, cdr)] from either OASIS-3 clinical CSV layout."""
    table: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        cdr = _pick(r, CDR_COLS)
        try:
            cdr = float(cdr)
        except (TypeError, ValueError):
            continue
        label = _pick(r, LABEL_COLS) or ""
        subject = _pick(r, SUBJECT_COLS) or (label.split("_")[0] if label else None)
        days = _pick(r, DAYS_COLS)
        try:
            days = int(float(days)) if days is not None else session_days(label)
        except ValueError:
            days = session_days(label)
        if subject and days is not None:
            table.setdefault(subject, []).append((days, cdr))
    return table


def cdr_to_label(cdr: float) -> int:
    return 0 if cdr == 0 else (1 if cdr <= 0.5 else 2)


def nearest_label(session: str, subject: str, table: dict, max_days: int):
    """(label, cdr, days_apart) for the CDR closest to the scan, or None."""
    days = session_days(session)
    visits = table.get(subject)
    if days is None or not visits:
        return None
    d, cdr = min(visits, key=lambda v: abs(v[0] - days))
    if abs(d - days) > max_days:
        return None
    return cdr_to_label(cdr), cdr, abs(d - days)


def _name_score(name: str) -> int:
    name = name.lower()
    return 3 if "cdr" in name else 2 if "b4" in name else 1 if "clinical" in name else 0


def _csvs_in(name: str, data: bytes) -> list[tuple[str, bytes]]:
    """(member name, bytes) for every CSV in a .csv / .zip / .tar(.gz) resource."""
    import tarfile
    import zipfile

    low = name.lower()
    if low.endswith(".csv"):
        return [(name, data)]
    out = []
    if low.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            out = [(m, z.read(m)) for m in z.namelist() if m.lower().endswith(".csv")]
    elif low.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            out = [(m.name, t.extractfile(m).read()) for m in t.getmembers()
                   if m.isfile() and m.name.lower().endswith(".csv")]
    return out


def _cdr_rows(data: bytes) -> list[dict] | None:
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig", errors="replace"))))
    return rows if rows and any(c in rows[0] for c in CDR_COLS) else None


def find_clinical_rows(x) -> list[dict]:
    """Find OASIS-3's CDR table among 0AS_data_files' resources (CSV, or inside
    a zip/tar archive). Prints what it saw when nothing matches."""
    files = x.subject_resources(DATA_FILES_SUBJECT)
    loose = sorted((f for f in files if f["name"].lower().endswith(".csv") and _name_score(f["name"])),
                   key=lambda f: -_name_score(f["name"]))
    archives = [f for f in files if f["name"].lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))]
    for f in loose + archives:
        members = _csvs_in(f["name"], x.download(f["uri"]))
        for member, data in sorted(members, key=lambda m: -_name_score(m[0])):
            rows = _cdr_rows(data)
            if rows:
                print(f"clinical CDR table: {f['experiment']}/{f['name']}"
                      + (f" -> {member}" if member != f["name"] else "") + f" ({len(rows)} rows)")
                return rows
    print(f"Files seen on {DATA_FILES_SUBJECT} ({len(files)}):")
    for f in files[:30]:
        print(f"   {f['experiment']}/{f['name']}")
    if len(files) > 30:
        print(f"   ... and {len(files) - 30} more")
    raise SystemExit(
        f"Could not find a CDR table among {DATA_FILES_SUBJECT}'s files (list above). Download the "
        "OASIS-3 clinical CDR CSV (UDS b4 / ADRC clinical data) and pass --clinical-csv path/to/file.csv"
    )


def main() -> None:
    from oasis_xnat import AccessDenied, OasisXnat

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--clinical-csv", default=None, help="local OASIS-3 CDR CSV (skips the lookup)")
    ap.add_argument("--max-days", type=int, default=365)
    ap.add_argument("--workers", type=int, default=4, help="parallel scan listings (be gentle)")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    cache_path = processed / "oasis3_scan_index.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    x = OasisXnat()
    x.login()
    sessions = [s for s in x.mr_sessions() if s["subject"] != DATA_FILES_SUBJECT]
    todo = [s for s in sessions if s["label"] not in cache]
    print(f"{len(sessions)} MR sessions in OASIS3; listing scans for {len(todo)} not yet cached")

    def list_one(s):
        try:
            return s["label"], [{"ID": r.get("ID"), "type": r.get("type", "")}
                                for r in x.scans(s["subject"], s["label"])]
        except AccessDenied:
            raise
        except Exception as exc:
            print(f"  could not list {s['label']}: {exc}")
            return s["label"], None

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for n, (label, scans) in enumerate(pool.map(list_one, todo), 1):
            if scans is not None:
                cache[label] = scans
            if n % 100 == 0:
                cache_path.write_text(json.dumps(cache))
                print(f"  {n}/{len(todo)} listed")
    cache_path.write_text(json.dumps(cache))

    if args.clinical_csv:
        rows = list(csv.DictReader(open(Path(args.clinical_csv).expanduser(), encoding="utf-8-sig")))
    else:
        rows = find_clinical_rows(x)
    table = parse_cdr_table(rows)

    out_sessions, out_labels, dropped = [], [], 0
    for s in sessions:
        scans = cache.get(s["label"]) or []
        bold = [sc["ID"] for sc in scans if is_rest_bold(sc["type"])]
        if not bold:
            continue
        hit = nearest_label(s["label"], s["subject"], table, args.max_days)
        if hit is None:
            dropped += 1
            continue
        label, cdr, apart = hit
        dwi = [sc["ID"] for sc in scans if is_dwi(sc["type"])]
        out_sessions.append({"experiment_id": s["label"], "subject_id": s["subject"],
                             "bold_ids": " ".join(bold), "dwi_ids": " ".join(dwi)})
        out_labels.append({"mr_session": s["label"], "label": label, "cdr": cdr, "days_apart": apart})

    for name, rows_out in (("graph_sessions.csv", out_sessions), ("graph_session_labels.csv", out_labels)):
        with open(processed / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]) if rows_out else ["experiment_id"])
            w.writeheader()
            w.writerows(rows_out)
    counts = {c: sum(r["label"] == c for r in out_labels) for c in (0, 1, 2)}
    subjects = len({r["subject_id"] for r in out_sessions})
    print(f"\n{len(out_sessions)} sessions with resting-state BOLD and a CDR within {args.max_days} days "
          f"({subjects} subjects; {sum(bool(r['dwi_ids']) for r in out_sessions)} also have DWI); "
          f"{dropped} dropped for no nearby CDR")
    print(f"Normal {counts[0]} · MCI {counts[1]} · Dementia {counts[2]}")
    print(f"written: {processed / 'graph_sessions.csv'}, {processed / 'graph_session_labels.csv'}")
    print("Next: python ml/graph/download_oasis_func.py --data-root", root, "--dry-run")


if __name__ == "__main__":
    main()
