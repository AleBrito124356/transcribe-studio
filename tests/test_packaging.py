"""Packaging: the wheel must install one namespaced package, nothing generic.

Earlier versions shipped top-level ``src`` and ``cli`` modules into
site-packages, which collide with any other project that does the same. This
test builds the real wheel (offline, no build isolation) and inspects it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_consistent():
    import transcribe_studio

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{transcribe_studio.__version__}"' in pyproject


def test_wheel_contains_only_the_namespaced_package(tmp_path):
    pytest.importorskip("setuptools", minversion="77")
    # Build from a copy: setuptools writes build/ and *.egg-info next to the
    # sources, and a test run should not leave those in the checkout.
    src = tmp_path / "src"
    src.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, src / name)
    shutil.copytree(
        ROOT / "transcribe_studio",
        src / "transcribe_studio",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    proc = subprocess.run(
        [
            sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
            "--no-index", "-q", str(src), "-w", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    if proc.returncode != 0:
        pytest.skip(f"could not build a wheel offline here: {proc.stderr[-300:]}")
    wheels = list(tmp_path.glob("transcribe_studio-*.whl"))
    assert len(wheels) == 1
    names = zipfile.ZipFile(wheels[0]).namelist()

    top_level = {n.split("/")[0] for n in names}
    assert {t for t in top_level if not t.endswith(".dist-info")} == {"transcribe_studio"}
    assert not any(n.startswith(("src/", "cli.py", "tests/")) for n in names)
    assert "transcribe_studio/__main__.py" in names

    entry_points = next(n for n in names if n.endswith("entry_points.txt"))
    text = zipfile.ZipFile(wheels[0]).read(entry_points).decode("utf-8")
    assert "transcribe-studio = transcribe_studio.cli:main" in text
