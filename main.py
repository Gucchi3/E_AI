from __future__ import annotations

import argparse
from collections.abc import Callable

from utils.config import AppConfig, load_config
from utils.workflows import run_train


WORKFLOWS: dict[str, Callable[[AppConfig], None]] = {
    "train": run_train,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny CIFAR-10 CNN.")
    parser.add_argument("--config", default="config.json", help="Path to a JSON configuration file (default: config.json).")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    WORKFLOWS[config.run.mode](config)


if __name__ == "__main__":
    main()
