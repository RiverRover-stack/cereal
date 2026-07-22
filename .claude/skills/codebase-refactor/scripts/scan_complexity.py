#!/usr/bin/env python3
"""
Scan a repo for refactor hotspots without needing Claude to read every file.

Flags:
  - files over LINE_THRESHOLD lines
  - functions/methods over FUNC_THRESHOLD lines (heuristic, indent/brace based)
  - nesting deeper than NEST_THRESHOLD levels

Uses radon (if installed) for real cyclomatic complexity on Python files as a bonus signal.
Falls back to pure heuristics everywhere else so it works with zero extra installs.

Usage:
  python3 scan_complexity.py <repo_path> [--exclude PATTERN ...] [--only FOLDER] [--json-out PATH]
"""
import argparse
import json
import os
import re
import subprocess
import sys

LINE_THRESHOLD = 400
FUNC_THRESHOLD = 50
NEST_THRESHOLD = 4

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".c", ".h", ".cpp",
    ".hpp", ".cc", ".rb", ".rs", ".php", ".cs", ".kt", ".swift",
}

DEFAULT_EXCLUDES = [
    "node_modules", "vendor", ".git", "dist", "build", "__pycache__",
    ".venv", "venv", "target", ".refactor",
]

FUNC_DEF_RE = re.compile(
    r"^\s*(def\s+\w+\s*\(|function\s+\w+\s*\(|async\s+function\s+\w+\s*\(|"
    r"(public|private|protected|static|\s)*\s*[\w<>\[\],\s]+\s+\w+\s*\([^;]*\)\s*\{|"
    r"func\s+\w+\s*\(|\w+\s*:\s*function\s*\(|\w+\s*=\s*\([^)]*\)\s*=>)"
)


def should_exclude(path, excludes):
    parts = path.split(os.sep)
    return any(ex in parts for ex in excludes)


def iter_code_files(repo, excludes, only=None):
    root = os.path.join(repo, only) if only else repo
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excludes and not d.startswith(".")]
        for fn in filenames:
            ext = os.path.splitext(fn)[1]
            if ext in CODE_EXTENSIONS:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, repo)
                if not should_exclude(rel, excludes):
                    yield full, rel


def max_nesting(lines, is_brace_lang):
    """
    Nesting depth that ignores continuation-line indentation (e.g. aligned
    multi-line function args/dict literals), which is not real block nesting.

    Brace languages (JS/Go/Java/C/...): track { } depth directly.
    Python-like (colon-block) languages: only increase depth when a logical
    statement line ends with ':' (a real block opener), using an indent
    stack so wrapped/aligned lines that don't open a block are ignored.
    """
    if is_brace_lang:
        depth = 0
        max_depth = 0
        for line in lines:
            code = line.split("//")[0]
            depth += code.count("{") - code.count("}")
            max_depth = max(max_depth, depth)
        return max_depth

    # Colon-block heuristic (Python, Ruby-ish)
    stack = []  # indent levels that opened a block
    max_depth = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        while stack and indent <= stack[-1]:
            stack.pop()
        max_depth = max(max_depth, len(stack))
        # Only a real block opener increases depth going forward — a line ending in
        # ':' that isn't inside brackets. Skip obvious dict/slice cases heuristically
        # by requiring the line to start a statement keyword or def/class, or simply
        # end with ':' with balanced brackets on the line.
        if stripped.endswith(":") and stripped.count("(") <= stripped.count(")") \
                and stripped.count("[") <= stripped.count("]") \
                and stripped.count("{") <= stripped.count("}"):
            stack.append(indent)
    return max_depth


def find_long_functions(lines):
    """Heuristic: find function-def lines, measure length until next def at same/lower indent."""
    long_funcs = []
    starts = []
    for i, line in enumerate(lines):
        if FUNC_DEF_RE.match(line):
            indent = len(line) - len(line.lstrip(" \t"))
            starts.append((i, indent, line.strip()[:60]))
    for idx, (start, indent, sig) in enumerate(starts):
        end = len(lines)
        for j in range(start + 1, len(lines)):
            nxt = lines[j]
            if nxt.strip() == "":
                continue
            nxt_indent = len(nxt) - len(nxt.lstrip(" \t"))
            if nxt_indent <= indent and FUNC_DEF_RE.match(nxt):
                end = j
                break
        length = end - start
        if length > FUNC_THRESHOLD:
            long_funcs.append({"start_line": start + 1, "length": length, "signature": sig})
    return long_funcs


def radon_complexity(repo):
    """Bonus signal for Python if radon is installed. Returns dict path->avg complexity, or {}."""
    try:
        out = subprocess.run(
            ["radon", "cc", repo, "-j"], capture_output=True, text=True, timeout=60
        )
        if out.returncode != 0:
            return {}
        data = json.loads(out.stdout)
        result = {}
        for path, blocks in data.items():
            if blocks:
                avg = sum(b["complexity"] for b in blocks) / len(blocks)
                result[os.path.relpath(path, repo)] = round(avg, 1)
        return result
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_path")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--only", default=None, help="Limit scan to one subfolder (for re-scans)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    excludes = set(DEFAULT_EXCLUDES) | set(args.exclude)
    repo = os.path.abspath(args.repo_path)

    hotspots = []
    total_files = 0
    for full, rel in iter_code_files(repo, excludes, args.only):
        total_files += 1
        try:
            with open(full, "r", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            continue
        n = len(lines)
        long_funcs = find_long_functions(lines)
        ext = os.path.splitext(rel)[1]
        is_brace_lang = ext not in {".py", ".rb"}
        nesting = max_nesting(lines, is_brace_lang)
        flags = []
        if n > LINE_THRESHOLD:
            flags.append(f"file_too_long({n} lines)")
        if long_funcs:
            flags.append(f"{len(long_funcs)}_long_function(s)")
        if nesting > NEST_THRESHOLD:
            flags.append(f"deep_nesting({nesting})")
        if flags:
            hotspots.append({
                "file": rel,
                "lines": n,
                "long_functions": long_funcs,
                "max_nesting": nesting,
                "flags": flags,
            })

    complexity_bonus = radon_complexity(repo)
    for h in hotspots:
        if h["file"] in complexity_bonus:
            h["avg_cyclomatic_complexity"] = complexity_bonus[h["file"]]

    out_path = args.json_out or os.path.join(repo, ".refactor", "scan_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"total_files_scanned": total_files, "hotspots": hotspots}, f, indent=2)

    print(f"Scanned {total_files} code files.")
    print(f"Flagged {len(hotspots)} hotspot files.")
    for h in sorted(hotspots, key=lambda x: -x["lines"])[:20]:
        print(f"  {h['file']}  ({h['lines']} lines)  {', '.join(h['flags'])}")
    print(f"Full report: {out_path}")


if __name__ == "__main__":
    main()
