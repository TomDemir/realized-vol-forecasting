#!/usr/bin/env python3
"""Download monthly BTCUSDT 1m klines from Binance Vision with SHA-256 verification.

Sequential downloads, no parallelism. Any checksum mismatch or network error
aborts the run with a non-zero exit code. Files are stored in data/raw/ (git-ignored).

Usage:
    python scripts/download.py [--start 2018-01] [--end 2026-08] [--out data/raw]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m"
DEFAULT_START = "2018-01"
DEFAULT_END = "2026-08"
RETRIES = 3
RETRY_SLEEP_S = 5.0


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings from start to end."""
    y0, m0 = (int(x) for x in start.split("-"))
    y1, m1 = (int(x) for x in end.split("-"))
    if (y0, m0) > (y1, m1):
        raise ValueError(f"start {start} is after end {end}")
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checksum_file(text: str, expected_name: str) -> str:
    """Parse a Binance .CHECKSUM file ('<sha256>  <filename>')."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].strip() == expected_name:
            return parts[0].strip().lower()
    # Fallback: single-line file with only the hash
    parts = text.split()
    if len(parts) >= 1 and len(parts[0]) == 64:
        return parts[0].lower()
    raise ValueError(f"cannot parse checksum file content: {text!r}")


def fetch(url: str, dest: Path) -> None:
    """Download url to dest with a small number of retries on network errors."""
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "realized-vol-forecasting/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                break  # not going to appear on retry
            if attempt < RETRIES:
                time.sleep(RETRY_SLEEP_S * attempt)
    if dest.exists():
        dest.unlink()
    raise RuntimeError(f"download failed for {url}: {last_err!r}")


def download_month(month: str, out_dir: Path) -> Path:
    name = f"BTCUSDT-1m-{month}.zip"
    zip_path = out_dir / name
    sum_path = out_dir / (name + ".CHECKSUM")

    # Always (re)fetch the checksum: it is tiny and authoritative.
    fetch(f"{BASE_URL}/{name}.CHECKSUM", sum_path)
    expected = parse_checksum_file(sum_path.read_text(), name)

    if zip_path.exists() and sha256_of(zip_path) == expected:
        print(f"[skip] {name} already present, checksum OK")
        return zip_path

    fetch(f"{BASE_URL}/{name}", zip_path)
    actual = sha256_of(zip_path)
    if actual != expected:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"CHECKSUM MISMATCH for {name}: expected {expected}, got {actual}. File removed."
        )
    print(f"[ok]   {name} ({zip_path.stat().st_size / 1e6:.2f} MB) sha256={actual[:12]}...")
    return zip_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--out", default="data/raw")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    months = month_range(args.start, args.end)
    print(f"Downloading {len(months)} months ({months[0]} .. {months[-1]}) into {out_dir}/")
    for month in months:
        try:
            download_month(month, out_dir)
        except Exception as e:  # noqa: BLE001 - we want to stop on anything
            print(f"\nFATAL at {month}: {e}", file=sys.stderr)
            return 1
    print("All months downloaded and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
