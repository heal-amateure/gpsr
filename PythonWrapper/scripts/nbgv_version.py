"""Derive the Python package version from Nerdbank.GitVersioning (version.json).

NuGet and PyPI use different version syntaxes, so the nbgv version is translated
to its PEP 440 equivalent:

    nbgv/NuGet               PEP 440
    1.0.2                 -> 1.0.2
    1.0.2-alpha           -> 1.0.2a0
    1.0.2-beta.3          -> 1.0.2b3
    1.0.2-rc.1            -> 1.0.2rc1
    non-public release    -> 1.0.2a0+g<commit>   (local version, never published)

Usage (stdlib only, needs `nbgv` or the `dotnet nbgv` tool on PATH):

    python scripts/nbgv_version.py           # print version
    python scripts/nbgv_version.py --apply   # also write it into pyproject.toml
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
_PRERELEASE_LABELS = {"alpha": "a", "a": "a", "beta": "b", "b": "b", "rc": "rc", "pre": "rc"}


def nbgv_json() -> dict:
    cmd = ["nbgv"] if shutil.which("nbgv") else ["dotnet", "nbgv"]
    out = subprocess.run(
        [*cmd, "get-version", "-f", "json"],
        cwd=PROJECT_DIR, check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def to_pep440(info: dict) -> str:
    version = info["SimpleVersion"]

    prerelease = info.get("PrereleaseVersion", "").lstrip("-")
    if prerelease:
        m = re.fullmatch(r"([A-Za-z]+)(?:\.(\d+))?", prerelease)
        if not m or m.group(1).lower() not in _PRERELEASE_LABELS:
            raise ValueError(f"Cannot map prerelease tag '-{prerelease}' to PEP 440")
        version += _PRERELEASE_LABELS[m.group(1).lower()] + (m.group(2) or "0")

    if not info["PublicRelease"]:
        version += f"+g{info['GitCommitIdShort']}"
    return version


def apply(version: str) -> None:
    pyproject = PROJECT_DIR / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    new, n = re.subn(r'(?m)^version = ".*"$', f'version = "{version}"', text, count=1)
    if n != 1:
        raise RuntimeError("No 'version = \"...\"' line found in pyproject.toml")
    pyproject.write_text(new, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the version into pyproject.toml")
    args = parser.parse_args()

    version = to_pep440(nbgv_json())
    if args.apply:
        apply(version)
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
