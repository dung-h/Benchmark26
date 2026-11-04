#!/usr/bin/env python3
"""Fetch pinned original files for review; never build features or run circuits."""

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["all", "mali", "qonductor", "qpack"], required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--asset", help="Optional exact upstream path, for a small download check")
    parser.add_argument("--list", action="store_true", help="Print selection; do not download")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "upstream/data_sources.json").read_text())
    assets = [
        a
        for a in manifest["assets"]
        if args.source in ("all", a["source"]) and (args.asset is None or a["path"] == args.asset)
    ]
    if not assets:
        parser.error("No matching manifest asset")
    if args.list:
        print(
            json.dumps(
                {
                    "files": len(assets),
                    "bytes": sum(a["bytes"] for a in assets),
                    "assets": [a["source"] + "/" + a["path"] for a in assets],
                },
                indent=2,
            )
        )
        return
    # Refuse destinations inside the scientific/review checkout.
    destination = args.destination.resolve()
    if destination == ROOT or ROOT in destination.parents:
        parser.error("Choose a destination outside the review checkout")
    records = []
    for a in assets:
        relative = Path(a["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        target = destination / a["source"] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != a["bytes"] or sha(target) != a["sha256"]:
                raise RuntimeError("Existing file differs; refusing overwrite: " + str(target))
            status = "EXISTING_HASH_MATCH"
        else:
            request = urllib.request.Request(a["url"], headers={"User-Agent": "QRE-private-review/1.0"})
            with tempfile.TemporaryDirectory(prefix=".qre-fetch-", dir=target.parent) as tmp:
                pending = Path(tmp) / "download"
                with urllib.request.urlopen(request, timeout=90) as response, pending.open("xb") as out:
                    while block := response.read(1024 * 1024):
                        out.write(block)
                if pending.stat().st_size != a["bytes"] or sha(pending) != a["sha256"]:
                    raise RuntimeError("Downloaded bytes differ (possibly an LFS pointer or error page): " + a["path"])
                # Exclusive create protects existing source files even during a concurrent attempt.
                with pending.open("rb") as inp, target.open("xb") as out:
                    while block := inp.read(1024 * 1024):
                        out.write(block)
            status = "DOWNLOADED_HASH_MATCH"
        records.append({"source": a["source"], "path": a["path"], "status": status, "sha256": a["sha256"]})
        print(json.dumps(records[-1]), flush=True)
    print(
        json.dumps(
            {
                "status": "PASS_SELECTED_DOWNLOADS",
                "files": len(records),
                "bytes": sum(a["bytes"] for a in assets),
                "destination": str(destination),
                "scope": "Original file acquisition and byte verification only; not cohort reconstruction or model training.",
            }
        )
    )


if __name__ == "__main__":
    main()
