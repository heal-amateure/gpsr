"""Loads the .NET runtime and the GPSR assemblies via pythonnet.

The DLLs are looked up in this order:
1. the directory given by the environment variable ``GPSR_LIB_DIR``
2. the ``lib`` folder inside this package (filled by ``dotnet publish``, see README)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ASSEMBLIES = ("GPSR.Utils", "GPSR.Data", "GPSR.Core")
_loaded = False


def lib_dir() -> Path:
    env = os.environ.get("GPSR_LIB_DIR")
    path = Path(env) if env else Path(__file__).parent / "lib"
    if not (path / "GPSR.Core.dll").is_file():
        raise FileNotFoundError(
            f"GPSR.Core.dll not found in '{path}'. Build it with\n"
            "  dotnet publish src/PGP/Core.csproj -c Release -o PythonWrapper/src/gpsr/lib\n"
            "or point GPSR_LIB_DIR to a folder containing the GPSR DLLs."
        )
    return path


def load() -> None:
    """Start the CoreCLR runtime (once) and reference the GPSR assemblies."""
    global _loaded
    if _loaded:
        return

    path = lib_dir()

    # Must happen before the first `import clr`; picks the newest installed .NET runtime.
    import pythonnet

    if pythonnet.get_runtime_info() is None:
        pythonnet.load("coreclr")

    import clr

    sys.path.append(str(path))
    for name in _ASSEMBLIES:
        clr.AddReference(name)

    # Format numbers in expressions as "0.96" instead of the system locale's "0,96".
    from System.Globalization import CultureInfo

    CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture
    CultureInfo.CurrentCulture = CultureInfo.InvariantCulture

    _loaded = True
