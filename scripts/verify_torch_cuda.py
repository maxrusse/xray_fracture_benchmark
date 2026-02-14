from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"torch_version={torch.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"device_0={torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
