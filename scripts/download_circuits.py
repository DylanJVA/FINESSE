"""
Download benchmark circuits from red-queen and/or QASMBench into circuits/.

After `pip install -e .`:
    finesse-download                      # both sources
    finesse-download --source redqueen
    finesse-download --source qasmbench

Already-downloaded files are skipped. Re-run to pick up new circuits upstream.

MQT Bench circuits are generated programmatically via the mqt-bench package
(optional dependency) — no download needed.
"""
import argparse
import json
import os
import sys
import urllib.request

# ── Source URLs ───────────────────────────────────────────────────────────────

# The red-queen repo reorganized and removed the misc/ benchmark directory.
# We use the last commit that contained these circuits as a stable archive.
_REDQUEEN_ARCHIVE_SHA = "c3e3e710"
_REDQUEEN_API = (
    f"https://api.github.com/repos/Qiskit/red-queen/git/trees/"
    f"{_REDQUEEN_ARCHIVE_SHA}?recursive=1"
)
_REDQUEEN_MISC_PREFIX = "red_queen/games/mapping/benchmarks/misc/"
_REDQUEEN_RAW = (
    f"https://raw.githubusercontent.com/Qiskit/red-queen/"
    f"{_REDQUEEN_ARCHIVE_SHA}/red_queen/games/mapping/benchmarks/misc/"
)

_QASMBENCH_API  = "https://api.github.com/repos/pnnl/QASMBench/contents/{size}?ref=master"
_QASMBENCH_RAW  = "https://raw.githubusercontent.com/pnnl/QASMBench/master/{size}/{name}/{name}.qasm"
_QASMBENCH_SIZES = ["small", "medium"]

# ── Local directories (mirrors finesse/benchmarks.py) ─────────────────────────

_REPO_ROOT     = os.path.dirname(os.path.dirname(__file__))
_REDQUEEN_DIR  = os.path.join(_REPO_ROOT, "circuits", "redqueen")
_QASMBENCH_DIR = os.path.join(_REPO_ROOT, "circuits", "qasmbench")

# ── Downloaders ───────────────────────────────────────────────────────────────

def download_redqueen() -> tuple[int, int, int]:
    os.makedirs(_REDQUEEN_DIR, exist_ok=True)
    print(f"red-queen: fetching circuit list from archive {_REDQUEEN_ARCHIVE_SHA[:8]}...", flush=True)
    with urllib.request.urlopen(_REDQUEEN_API) as r:
        tree = json.loads(r.read())
    qasms = sorted(
        item["path"].removeprefix(_REDQUEEN_MISC_PREFIX)
        for item in tree["tree"]
        if item["path"].startswith(_REDQUEEN_MISC_PREFIX)
        and item["path"].endswith(".qasm")
    )
    print(f"  {len(qasms)} circuits found.", flush=True)

    downloaded = skipped = failed = 0
    for i, name in enumerate(qasms):
        dest = os.path.join(_REDQUEEN_DIR, name)
        if os.path.exists(dest):
            skipped += 1
            continue
        try:
            with urllib.request.urlopen(_REDQUEEN_RAW + name) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
            print(f"  [{i+1:3}/{len(qasms)}] {name:<40} {len(data)//1024:>4} KB", flush=True)
            downloaded += 1
        except Exception as e:
            print(f"  [{i+1:3}/{len(qasms)}] {name:<40} FAILED: {e}", flush=True)
            failed += 1

    return downloaded, skipped, failed


def download_qasmbench() -> tuple[int, int, int]:
    downloaded = skipped = failed = 0
    for size in _QASMBENCH_SIZES:
        dest_dir = os.path.join(_QASMBENCH_DIR, size)
        os.makedirs(dest_dir, exist_ok=True)

        print(f"qasmbench/{size}: fetching circuit list...", flush=True)
        with urllib.request.urlopen(_QASMBENCH_API.format(size=size)) as r:
            items = json.loads(r.read())
        circuit_names = sorted(i["name"] for i in items if i["type"] == "dir")
        print(f"  {len(circuit_names)} circuits found.", flush=True)

        for i, name in enumerate(circuit_names):
            dest = os.path.join(dest_dir, name, f"{name}.qasm")
            if os.path.exists(dest):
                skipped += 1
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            url = _QASMBENCH_RAW.format(size=size, name=name)
            try:
                with urllib.request.urlopen(url) as r:
                    data = r.read()
                with open(dest, "wb") as f:
                    f.write(data)
                print(f"  [{i+1:3}/{len(circuit_names)}] {name:<40} {len(data)//1024:>4} KB", flush=True)
                downloaded += 1
            except Exception as e:
                print(f"  [{i+1:3}/{len(circuit_names)}] {name:<40} FAILED: {e}", flush=True)
                failed += 1

    return downloaded, skipped, failed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["redqueen", "qasmbench", "all"],
        default="all",
        help="which circuit suite to download (default: all)",
    )
    args = parser.parse_args()

    total_dl = total_skip = total_fail = 0

    if args.source in ("redqueen", "all"):
        dl, sk, fa = download_redqueen()
        total_dl += dl; total_skip += sk; total_fail += fa
        print()

    if args.source in ("qasmbench", "all"):
        dl, sk, fa = download_qasmbench()
        total_dl += dl; total_skip += sk; total_fail += fa
        print()

    print(f"Done. Downloaded: {total_dl}  Cached: {total_skip}  Failed: {total_fail}")
    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
