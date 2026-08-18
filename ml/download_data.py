"""Download and normalise the DementiaBank Pitt 'Cookie Theft' transcripts.

Source: https://huggingface.co/datasets/MearaHe/dementiabank (ungated mirror)
498 picture-description transcripts labelled dementia / control.

Writes ml/data/transcripts.csv with columns: text, label (1 = dementia, 0 = control).
"""

import csv
import json
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/MearaHe/dementiabank/"
    "resolve/main/Updated_Formatted_Dataset.json"
)

DATA_DIR = Path(__file__).parent / "data"
RAW_FILE = DATA_DIR / "dementiabank_raw.json"
OUT_FILE = DATA_DIR / "transcripts.csv"

LABEL_MAP = {"dementia": 1, "control": 0}


def download() -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists():
        print(f"Using cached {RAW_FILE}")
        return json.loads(RAW_FILE.read_text())

    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
    RAW_FILE.write_text(raw)
    print(f"Saved {RAW_FILE} ({len(raw):,} bytes)")
    return json.loads(raw)


def normalise(records: list[dict]) -> list[dict]:
    rows, skipped = [], 0
    for r in records:
        text = (r.get("input") or "").strip()
        label = LABEL_MAP.get((r.get("output") or "").strip().lower())
        # Very short samples carry no usable linguistic signal.
        if label is None or len(text.split()) < 10:
            skipped += 1
            continue
        rows.append({"text": text, "label": label})
    if skipped:
        print(f"Skipped {skipped} unusable rows")
    return rows


def main() -> int:
    rows = normalise(download())
    if not rows:
        print("ERROR: no usable rows parsed", file=sys.stderr)
        return 1

    with OUT_FILE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    n_dem = sum(r["label"] for r in rows)
    print(f"\nWrote {OUT_FILE}")
    print(f"  total    : {len(rows)}")
    print(f"  dementia : {n_dem}")
    print(f"  control  : {len(rows) - n_dem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
