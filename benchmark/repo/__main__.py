"""Entry: from helium_demo/benchmark with PYTHONPATH=../src:.

  python -m repo              # full experiment run (needs free GPUs)
  python -m repo --dry-run    # dataset + strategy plan only (no vLLM)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_paths() -> None:
    # here:repo 
    here = Path(__file__).resolve().parent
    # benchmark_dir: helium_demo/benchmark
    benchmark_dir = here.parent
    # helium_demo: helium_demo
    helium_demo = benchmark_dir.parent
    # src: helium_demo/src
    src = helium_demo / "src"
    # add src and benchmark_dir to sys.path
    for path in (str(src), str(benchmark_dir)):
        if path not in sys.path:
            sys.path.insert(0, path) # insert at the beginning of the path


def main(argv: list[str] | None = None) -> None:
    _ensure_paths()
    parser = argparse.ArgumentParser(description="Experiment-group scheduler")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build dataset and print schedule plan without starting vLLM",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        from repo.scheduler import dry_run_plan

        plan = dry_run_plan()
        for line in plan.lines():
            print(line, flush=True)
        return

    from repo import config as cfg
    from repo.scheduler import run_experiment
    from repo.strategy import AllSingleCardStrategy

    # 实际在这里运行
    report = run_experiment(strategy=AllSingleCardStrategy())
    for line in report.lines():
        print(line, flush=True)
    print(f"wrote {cfg.RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
