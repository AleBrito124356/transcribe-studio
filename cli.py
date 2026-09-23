#!/usr/bin/env python3
"""Source-checkout entry point: ``python cli.py <command> ...``.

The real CLI lives in :mod:`transcribe_studio.cli`. This shim keeps every
command in the README working from a clone without installing anything; after
``pip install .`` the same CLI is available as ``transcribe-studio`` and as
``python -m transcribe_studio``.
"""

from transcribe_studio.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
