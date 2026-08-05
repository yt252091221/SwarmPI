"""
One-click reproduction entrypoint.

Runs everything from scratch. Results saved to handoff_bundle/.

Usage (from handoff_bundle root):
    python runall.py
    python runall.py --force
    python runall.py --ablation --force
    python code/reproduce_all.py --skip-run
    python code/reproduce_all.py --ablation
    python code/reproduce_all.py --table4

Environment variables (optional):
    TRACKEVAL_ROOT  — path to TrackEval (default: ../TrackEval relative to bundle)
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent.parent
CODE = BUNDLE / "code"
PYTHON = Path(sys.executable)


def run_script(script: str, extra_args: list[str] = None):
    cmd = [str(PYTHON), str(CODE / script)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, cwd=str(BUNDLE), check=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="One-click reproduction")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-run existing ablation and tracker outputs")
    parser.add_argument("--ablation", action="store_true",
                        help="Run the separate UAVSwarm-25 ablation only")
    parser.add_argument("--table4", action="store_true",
                        help="Run Table IV only (default)")
    args = parser.parse_args()

    if args.ablation and args.table4:
        parser.error("--ablation and --table4 are mutually exclusive")

    extra = []
    if args.skip_run:
        extra.append("--skip-run")
    if args.force:
        extra.append("--force")

    if args.ablation:
        print("\n" + "=" * 60)
        print("ABLATION: UAVSwarm-25")
        print("=" * 60)
        run_script("run_ablation_uavswarm25.py", extra)
        print(f"\n[DONE] Ablation: {BUNDLE / 'ablation_reproduction.md'}")
        return

    print("\n" + "=" * 60)
    print("TABLE IV: 6 methods x 6 datasets + COMBINED")
    print("=" * 60)
    run_script("run_table4_all_methods.py", extra)

    reference_checker = CODE / "compare_table4_reference.py"
    if reference_checker.exists():
        run_script("compare_table4_reference.py")

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print(f"  Table IV: {BUNDLE / 'table4_reproduction.md'}")
    print(f"  Reference diff: {BUNDLE / 'table4_reference_comparison.md'}")
    print("  Ablation: run `python runall.py --ablation --force` separately")
    print("=" * 60)


if __name__ == "__main__":
    main()
