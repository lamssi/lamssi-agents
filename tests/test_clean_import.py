"""A bare ``import lamssi_agents`` stays inert and dependency-light."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Modules that a bare package import must not load.
FORBIDDEN_MODULES = (
    "lamssi_agents.features.files.read",      # the file tool implementation
    "lamssi_agents.features.files.table",     # optional dataframe extraction
    "lamssi_packages.codecheck",          # a pack, not the kernel
)

# Optional dependencies that must remain lazy.
FORBIDDEN_THIRD_PARTY = (
    "docling",
    "pandas",
    "numpy",
    "PIL",
    "h5py",
    "nptdms",
    "requests",
    "litellm",
)

# The child reports modules loaded and paths touched after a bare import.
CHILD = textwrap.dedent(
    r"""
    import json, pathlib, sys

    probes = []
    _real = {
        name: getattr(pathlib.Path, name)
        for name in ("exists", "is_file", "is_dir", "iterdir", "glob", "rglob")
    }

    def _wrap(name, fn):
        def probe(self, *a, **kw):
            probes.append((name, str(self)))
            return fn(self, *a, **kw)
        return probe

    for name, fn in _real.items():
        setattr(pathlib.Path, name, _wrap(name, fn))

    import lamssi_agents

    for name, fn in _real.items():
        setattr(pathlib.Path, name, fn)

    PKG_ROOT = str(pathlib.Path(lamssi_agents.__file__).resolve().parent)
    outside = sorted({
        p for _, p in probes
        if not str(pathlib.Path(p).resolve()).startswith(PKG_ROOT)
    })

    print("@@RESULT@@" + json.dumps({
        "loaded": sorted(m for m in sys.modules if m.startswith("lamssi")),
        "third_party": sorted({m.split(".")[0] for m in sys.modules}),
        "outside_probes": outside[:40],
        "probe_count": len(probes),
    }))
    """
)

def _run_child(tmp_path: Path) -> dict:
    """Import the package from an empty cwd and report what it touched."""
    import json

    proc = subprocess.run(
        [sys.executable, "-c", CHILD],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **{k: v for k, v in __import__("os").environ.items() if not k.startswith("LAMSSI_")},
            "PYTHONPATH": str(REPO_ROOT),
        },
    )
    assert proc.returncode == 0, f"child failed:\n{proc.stdout}\n{proc.stderr}"
    marker = "@@RESULT@@"
    line = next(l for l in proc.stdout.splitlines() if l.startswith(marker))
    return json.loads(line[len(marker):])

def test_import_loads_no_domain_machinery(tmp_path: Path) -> None:
    result = _run_child(tmp_path)
    loaded = set(result["loaded"])
    offenders = sorted(m for m in FORBIDDEN_MODULES if m in loaded)
    assert not offenders, (
        "a name-only import pulled in machinery it should not have: "
        f"{offenders}"
    )

def test_import_pulls_no_optional_dependency(tmp_path: Path) -> None:
    """A bare package import does not load optional dependencies."""
    third_party = set(_run_child(tmp_path)["third_party"])
    offenders = sorted(m for m in FORBIDDEN_THIRD_PARTY if m in third_party)
    assert not offenders, f"a bare import loaded optional dependencies: {offenders}"

def test_import_touches_no_path_outside_the_package(tmp_path: Path) -> None:
    """Import must not make decisions about the embedding application's layout."""
    result = _run_child(tmp_path)
    assert not result["outside_probes"], (
        "import probed the filesystem outside the package:\n  "
        + "\n  ".join(result["outside_probes"])
    )
