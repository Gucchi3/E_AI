from __future__ import annotations

import argparse
from collections.abc import Callable

from utils.config import AppConfig, load_config
from utils.workflows import run_train


WORKFLOWS: dict[str, Callable[[AppConfig], None]] = {
    "train": run_train,
}


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を読み取る。"""
    parser = argparse.ArgumentParser(description="CIFAR-10 CNNを学習します。")
    parser.add_argument("--config", default="config/basic_cnn/cnn.json", help="設定ファイルのパス。既定値: config/basic_cnn/cnn.json")
    return parser.parse_args()


def main() -> None:
    """設定された処理を実行する。"""
    args   = parse_args()
    config = load_config(args.config)
    WORKFLOWS[config.run.mode](config)


if __name__ == "__main__":
    main()
