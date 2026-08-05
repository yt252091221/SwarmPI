"""Repository-root one-click entrypoint for the Table IV reproduction."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "code" / "reproduce_all.py"),
        run_name="__main__",
    )
