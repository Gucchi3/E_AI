# E_AI

CIFAR-10 用の小型 FP32 CNN を、設定ファイルで実行するための最小プロジェクトです。
すべての実行は `main.py` から開始します。

## セットアップ

Python 3.10 以上で、次をインストールしてください。

```powershell
pip install torch torchvision matplotlib rich pretty-errors
```

## 学習

初回は CIFAR-10 が `data/` にダウンロードされます。

```powershell
cd E_AI
python main.py
```

別設定を使う場合も、入口は同じです。

```powershell
python main.py --config my_config.json
```

主な設定は次のとおりです。

- `run.log_dir`: 実行成果物を保存する親ディレクトリ（既定値: `log`）
- `data.image_size`: `32` または `256`
- `data.normalization`: CIFAR-10 統計を使う `cifar10`、または `[0, 1]` の `zero_one`
- `data.batch_size`, `data.num_workers`
- `train.epochs`, `train.learning_rate`, `train.weight_decay`, `train.label_smoothing`

## 学習成果物

各実行時に `run.log_dir/YYYYMMDD_HHMMSS/` が作成され、次を保存します。

- `config.json`: 検証後に実際に使った設定
- `training_info.json`: device、PyTorch 版、モデル情報
- `training.log`: epoch ごとのログ
- `metrics.jsonl`: epoch ごとの機械可読な指標
- `curves.png`: train/test の loss・accuracy 曲線
- `model_best.pth`: test accuracy が最良だった時点の `state_dict`
- `model_final.pth`: 最終 epoch の `state_dict`

`.pth` はモデルの `state_dict` だけを保存する重みファイルです。optimizer 状態を含む
再開用 checkpoint、重み変換、Observer、量子化はまだ実装していません。
