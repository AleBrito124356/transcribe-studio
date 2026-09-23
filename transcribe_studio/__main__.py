"""Allow ``python -m transcribe_studio <command> ...``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main(prog="python -m transcribe_studio"))
