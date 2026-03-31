#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


EXPECTED_FILES = [
    "posterior_summary_hierarchical.csv",
    "market_channel_betas.csv",
    "pred_vs_actual_hierarchical.csv",
    "roi_by_market_channel.csv",
    "roi_global_channels.csv",
    "budget_recommendation_excellent.csv",
    "time_cv_results.csv",
    "time_cv_summary.csv",
    "time_cv_predictions.csv",
    "benchmark_lightweightmmm_status.csv",
    "benchmark_model_comparison.csv",
    "benchmark_predictions.csv",
]


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    docs_dir = project_root / "docs"
    cache_dir = project_root / ".cache"
    cache_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(cache_dir)
    env["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    env["PYTENSOR_FLAGS"] = f"base_compiledir={cache_dir / 'pytensor'}"
    env["PYTHONPYCACHEPREFIX"] = str(cache_dir / "pycache")

    cmd = [sys.executable, "src/main.py", "--quick"]
    print("Running smoke pipeline:", " ".join(cmd))
    subprocess.run(cmd, cwd=project_root, env=env, check=True)

    missing = []
    empty = []
    for filename in EXPECTED_FILES:
        path = docs_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        if path.stat().st_size == 0:
            empty.append(filename)

    if missing or empty:
        if missing:
            print("Missing files:")
            for f in missing:
                print(f"  - {f}")
        if empty:
            print("Empty files:")
            for f in empty:
                print(f"  - {f}")
        return 1

    print("Smoke test passed. All expected files are generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
