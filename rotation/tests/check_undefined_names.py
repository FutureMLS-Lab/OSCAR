#!/usr/bin/env python3
"""Fail on names that only blow up when their code path runs.

The mixed decode allocator called gpu_flush_pq_apply without importing it.
Python resolves a name inside a function body at call time, so the import
probe, 94 unit tests (which import the flush directly) and a five-model smoke
all passed, and the first request that reached a PQ flush killed the scheduler
with a NameError. pyflakes sees that statically in seconds. This script runs it
over the files given on the command line and exits non-zero on any undefined
name or undefined local; the image build feeds it every file that differs from
upstream.

Usage: python3 check_undefined_names.py FILE.py [FILE.py ...]
       (needs pyflakes: pip install pyflakes, or `uvx --from pyflakes python3`)

python/sglang/srt/arg_groups/fields/ is skipped: upstream writes option choices
as bare names inside annotations there, which never execute.
"""
import os
import sys

try:
    from pyflakes import api as pyflakes_api
    from pyflakes import messages as M
except ImportError:
    print("check_undefined_names: pyflakes is not installed; refusing to pass silently", file=sys.stderr)
    sys.exit(2)

SKIP_PREFIXES = ("python/sglang/srt/arg_groups/fields/",)
FAIL_ON = (M.UndefinedName, M.UndefinedLocal)


class Reporter:
    def __init__(self):
        self.hits = []
        self.errors = []

    def unexpectedError(self, filename, msg):
        self.errors.append(f"{filename}: {msg}")

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.errors.append(f"{filename}:{lineno}:{offset}: syntax error: {msg}")

    def flake(self, message):
        if isinstance(message, FAIL_ON):
            self.hits.append(
                f"{message.filename}:{message.lineno}:{message.col}: "
                + (message.message % message.message_args)
            )


def main(argv):
    files = [f for f in argv if f.endswith(".py")]
    if not files:
        print(__doc__, file=sys.stderr)
        return 2
    reporter = Reporter()
    scanned = 0
    for f in files:
        rel = os.path.relpath(f).replace(os.sep, "/")
        if rel.startswith(SKIP_PREFIXES) or not os.path.exists(f):
            continue  # deleted files show up in `git diff --name-only`
        pyflakes_api.checkPath(f, reporter)
        scanned += 1
    for line in reporter.errors + reporter.hits:
        print(line)
    status = "FAILED" if (reporter.hits or reporter.errors) else "clean"
    print(f"[undefined-names] {scanned} files scanned: {status}")
    return 1 if (reporter.hits or reporter.errors) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
