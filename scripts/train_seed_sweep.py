from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark.utils import load_yaml, save_json, save_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed training from one base config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", default="42,1337,2025")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _parse_seed_list(raw: str) -> list[int]:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not items:
        raise ValueError("--seeds cannot be empty")
    return [int(x) for x in items]


def main() -> int:
    args = parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(config_path)
    seeds = _parse_seed_list(args.seeds)

    results: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = output_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = run_dir / "best_model.pt"
        if args.skip_existing and best_ckpt.exists():
            results.append(
                {
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "status": "skipped_existing",
                }
            )
            continue

        cfg = dict(base_cfg)
        cfg["seed"] = int(seed)
        cfg_path = run_dir / "seed_config.yaml"
        save_yaml(cfg_path, cfg)

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--config",
            str(cfg_path),
            "--output-dir",
            str(run_dir),
        ]
        if args.init_checkpoint:
            cmd.extend(["--init-checkpoint", str((REPO_ROOT / args.init_checkpoint).resolve())])

        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        status = "ok" if completed.returncode == 0 else f"failed_{completed.returncode}"
        results.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "status": status,
                "returncode": int(completed.returncode),
            }
        )
        if completed.returncode != 0:
            break

    payload = {
        "base_config": str(config_path),
        "output_root": str(output_root),
        "seeds": seeds,
        "results": results,
    }
    save_json(output_root / "seed_sweep_summary.json", payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
