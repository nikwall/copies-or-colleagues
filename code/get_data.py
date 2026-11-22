"""Download and unpack the ForecastBench snapshot used in the paper (17 November 2026).

ForecastBench republishes processed_forecast_sets.tar.gz every night, so the copy on forecastbench.org
changes over time. The exact snapshot behind the paper is attached to release v1.0 of this repository.
This script downloads it, checks its SHA-256 and unpacks it to
data/raw/forecastbench/forecastbench-processed-forecast-sets/ (about 4.2 GB, 40 rounds).

Usage: python -I code/get_data.py <project folder>
"""
import hashlib
import os
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longpath import lp  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
URL = ("https://github.com/nikwall/copies-or-colleagues/releases/download/v1.0/"
       "processed_forecast_sets_2026-11-17.tar.gz")
SHA256 = "3b3b9630f2ccf4da401e703266a116cdc7de535eabb4cb44c9a10d1c01315b0f"
RAW = os.path.join(ROOT, "data", "raw", "forecastbench")
TAR = os.path.join(RAW, "processed_forecast_sets_2026-11-17.tar.gz")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


os.makedirs(RAW, exist_ok=True)
if not (os.path.exists(TAR) and sha256(TAR) == SHA256):
    print("downloading", URL)
    urllib.request.urlretrieve(URL, TAR)
digest = sha256(TAR)
if digest != SHA256:
    sys.exit(f"checksum mismatch: {digest}")
print("checksum ok")

n = 0
with tarfile.open(TAR, "r:gz") as tf:
    for m in tf:
        if not (m.isfile() or m.isdir()):
            continue
        name = m.name.replace("\\", "/")
        if name.startswith("/") or ".." in name.split("/"):
            sys.exit(f"unsafe member in archive: {name}")
        target = lp(os.path.join(RAW, *name.split("/")))
        if m.isdir():
            os.makedirs(target, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with tf.extractfile(m) as src, open(target, "wb") as dst:
            for b in iter(lambda: src.read(1 << 20), b""):
                dst.write(b)
        n += 1
print("unpacked", n, "files to", os.path.join(RAW, "forecastbench-processed-forecast-sets"))
