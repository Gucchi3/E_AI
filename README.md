# E_AI

Embedded AIの研究に必要な範囲へ絞った、CIFAR-10用の小さなFP32 CNN学習コードです。入口は`main.py`だけで、設定は`config.json`にまとめています。

## セットアップ

Python 3.10以上で、次をインストールしてください。

```powershell
pip install torch torchvision matplotlib rich pretty-errors
```

## 使い方

初回実行時はCIFAR-10が`data/`へ自動ダウンロードされます。

```powershell
python main.py
```

別の設定ファイルを使う場合も入口は同じです。

```powershell
python main.py --config my_config.json
```

主な設定は次のとおりです。

- `run.device`: `auto`、`cpu`、`cuda`
- `run.log_dir`: 実行結果を保存する親ディレクトリ
- `data.image_size`: `32`または`256`
- `data.normalization`: CIFAR-10統計を使う`cifar10`、または`[0, 1]`の`zero_one`
- `data.batch_size`、`data.num_workers`
- `train.epochs`、`train.learning_rate`、`train.weight_decay`、`train.label_smoothing`

## 表示

実行時の設定は、Q_ViTと同じ`rich.Table`の「Training Configuration」として表示されます。各epochはQ_ViTと同じ簡潔な形式です。

```text
Epoch [  1/20]  lr=1.00e-03  train_loss=1.2345  test_loss=1.1234  acc=60.00%
```

best weightの更新は画面へ出力しません。

## 出力

実行ごとに`run.log_dir/YYYYMMDD_HHMMSS/`を作り、次を保存します。

- `config.json`: 実際に使用した設定
- `training_info.json`: device、PyTorch、モデル、入力の基本情報
- `metrics.jsonl`: epochごとのlossとaccuracy
- `curves.png`: train/test lossとtest accuracyの曲線
- `model_best.pth`: test accuracyが最良だったモデルの`state_dict`
- `model_final.pth`: 最終epochのモデルの`state_dict`

独自のloggerや`training.log`は作りません。量子化、Observer、resume checkpointなどは、必要になるまで含めない方針です。
