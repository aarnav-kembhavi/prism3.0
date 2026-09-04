"""
Exercise a deployed PRISM service and compare its output with a local run.

    python deploy/client_check.py <url> parse   <file> [--out result.md]
    python deploy/client_check.py <url> compare <file> <local.md>
    python deploy/client_check.py <url> cold    <file>

`cold` times a request against an instance that has scaled to zero, so it
measures cold start plus one page rather than warm per-page latency.
"""
import sys
import time
import unicodedata
import re
from pathlib import Path

import requests


def _norm(s: str) -> str:
    """Same normalisation the sweep's comparison uses: NFKC + whitespace."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def parse(url: str, path: str, timeout: int = 900):
    t0 = time.perf_counter()
    with open(path, "rb") as fh:
        r = requests.post(f"{url}/parse", files={"file": (Path(path).name, fh)},
                          timeout=timeout)
    return r, time.perf_counter() - t0


def main() -> int:
    url, mode, path = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]

    r, wall = parse(url, path)
    print(f"POST /parse {Path(path).name}: http {r.status_code}, "
          f"{len(r.content)} bytes, {wall:.2f}s")
    if r.status_code != 200:
        print(r.text[:500])
        return 1

    remote = r.text
    out = None
    for i, a in enumerate(sys.argv):
        if a == "--out":
            out = sys.argv[i + 1]
    if out:
        Path(out).write_text(remote, encoding="utf-8")
        print("wrote", out)

    if mode == "compare":
        local = Path(sys.argv[4]).read_text(encoding="utf-8")
        same_bytes = remote == local
        same_norm = _norm(remote) == _norm(local)
        print(f"local  {len(local)} bytes")
        print(f"remote {len(remote)} bytes")
        print(f"identical (exact)      : {same_bytes}")
        print(f"identical (normalised) : {same_norm}")
        if not same_norm:
            try:
                from rapidfuzz.distance import Levenshtein
                sim = Levenshtein.normalized_similarity(_norm(local), _norm(remote))
                print(f"similarity             : {sim:.5f}")
            except ImportError:
                pass
            import difflib
            diff = list(difflib.unified_diff(
                local.splitlines(), remote.splitlines(),
                "local", "cloudrun", lineterm="", n=1))
            print("\n".join(diff[:40]))
        return 0 if same_norm else 2

    if mode == "cold":
        print(f"cold request (scaled to zero) total: {wall:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
