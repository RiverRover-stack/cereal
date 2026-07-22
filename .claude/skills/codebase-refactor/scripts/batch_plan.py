#!/usr/bin/env python3
"""
Turn .refactor/scan_report.json into an ordered batch plan grouped by top-level
folder/module, splitting any folder with too many hotspots into sub-batches.

Usage:
  python3 batch_plan.py <repo_path> [--max-per-batch N] [--scan-report PATH]
"""
import argparse
import json
import os
from collections import defaultdict

DEFAULT_MAX_PER_BATCH = 6


def top_module(rel_path, depth=2):
    parts = rel_path.split(os.sep)
    if len(parts) <= 1:
        return "(root)"
    return os.sep.join(parts[:depth]) if len(parts) > depth else parts[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_path")
    ap.add_argument("--max-per-batch", type=int, default=DEFAULT_MAX_PER_BATCH)
    ap.add_argument("--scan-report", default=None)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_path)
    scan_path = args.scan_report or os.path.join(repo, ".refactor", "scan_report.json")
    with open(scan_path) as f:
        scan = json.load(f)

    groups = defaultdict(list)
    for h in scan["hotspots"]:
        groups[top_module(h["file"])].append(h["file"])

    batches = []
    for module, files in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(files) <= args.max_per_batch:
            batches.append({"module": module, "files": files})
        else:
            # split into chunks of max_per_batch, divide and conquer
            for i in range(0, len(files), args.max_per_batch):
                chunk = files[i:i + args.max_per_batch]
                batches.append({
                    "module": f"{module} (part {i // args.max_per_batch + 1})",
                    "files": chunk,
                })

    out_path = os.path.join(repo, ".refactor", "batches.json")
    with open(out_path, "w") as f:
        json.dump({"batches": batches}, f, indent=2)

    print(f"Built {len(batches)} batch(es):")
    for i, b in enumerate(batches, 1):
        print(f"  {i}. {b['module']}  ({len(b['files'])} file(s))")
    print(f"Plan saved to: {out_path}")


if __name__ == "__main__":
    main()
