from __future__ import annotations

import subprocess
import sys

FORBIDDEN_PATTERNS = [
    "data/**",
    "results/**",
    "artifacts/**",
    "runs/**",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.7z",
    "*.dcm",
    "*.nii",
    "*.nii.gz",
]


def main() -> int:
    cmd = ["git", "ls-files", *FORBIDDEN_PATTERNS]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Failed to run git check. Is this a git repository?", file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        return 2

    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if files:
        print("Tracked dataset/artifact files are not allowed:")
        for path in files:
            print(path)
        return 1

    print("No forbidden tracked dataset/artifact files found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

