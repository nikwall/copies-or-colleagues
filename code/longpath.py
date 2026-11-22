r"""Open files whose absolute path exceeds Windows' 260-character limit.

Some ForecastBench file names are long enough that the full path in this project
folder passes the limit; the extended-length prefix \\?\ lifts it.
"""
import os


def lp(path):
    """Return an extended-length path on Windows, the path itself elsewhere."""
    if os.name != "nt":
        return path
    full = os.path.abspath(path).replace("/", "\\")
    return full if full.startswith("\\\\?\\") else "\\\\?\\" + full


if __name__ == "__main__":
    import sys
    d = sys.argv[1]
    fn = sorted(os.listdir(lp(d)))[0]
    p = lp(os.path.join(d, fn))
    print(len(p), os.path.exists(p), os.path.getsize(p))
    with open(p, encoding="utf-8") as f:
        print(f.read(300))
