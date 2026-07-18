# E_AI

Embedded AIの研究に必要な範囲へ絞った、CIFAR-10用の小さなCNN学習コードです。FP32学習と基礎的な整数QATに対応しています。入口は`main.py`だけで、設定はJSONへまとめています。

## セットアップ

Python 3.10以上で、次をインストールしてください。

```powershell
pip install torch torchvision matplotlib rich pretty-errors thop
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

小型QATモデルを学習する場合は、付属の設定を指定します。

```powershell
python main.py --config config_qat.json
```

主な設定は次のとおりです。

- `run.device`: `auto`、`cpu`、`cuda`
- `run.log_dir`: 実行結果を保存する親ディレクトリ
- `data.image_size`: `32`または`256`
- `data.normalization`: CIFAR-10統計を使う`cifar10`、または`[0, 1]`の`zero_one`
- `data.batch_size`、`data.num_workers`
- `model.load_weight`: 学習済み重みを読み込む場合は`true`
- `model.weight_path`: 読み込む`.pth`ファイルのパス
- `quantization.weight_bits`: 重みの整数bit数。`2`、`4`、`8`、`16`
- `quantization.activation_bits`: 活性の整数bit数。`2`、`4`、`8`、`16`
- `quantization.input_bits`: `uint8`入力に合わせて現在は`8`
- `quantization.rounding`: `ties_away_from_zero`、`ties_to_positive`、`ties_to_even`
- `train.epochs`、`train.learning_rate`、`train.weight_decay`、`train.label_smoothing`

学習済み重みを使う場合は、`config.json`を次のように設定します。

```json
"model": {
  "name"        : "cifar_cnn",
  "num_classes" : 10,
  "load_weight" : true,
  "weight_path" : "log/20260718_120000/model_best.pth"
}
```

raw `state_dict`、`model`キーを持つcheckpoint、`state_dict`キーを持つcheckpointを読み込めます。モデル構造と重みの構造は一致している必要があります。

## QAT

`qat_cifar_cnn`は、重みを出力チャネル単位、ReLU後の活性をTensor単位でFake Quantizationします。丸めの既定値は`ties_away_from_zero`です。入力画像は保存形式と実機では0～255の`uint8`とし、PyTorch内では`ToTensor()`後の0～1へ`scale=1/255`を適用して同じ整数値を模擬します。

量子化部品は`utils/quantization/`内で完結しており、PyTorch以外のプロジェクト内moduleへ依存しません。scaleは各`IntegerQuantizer`の`scale`、整数値は`quantize()`の結果から取得できます。現在は基礎実装に絞り、Observer、freeze/unfreeze、BatchNorm fold、FP4は含めていません。

## 表示

実行時の設定は、Q_ViTと同じ`rich.Table`の「Training Configuration」として表示されます。パラメータ数に加えて、1枚の画像を入力したときのMACsも表示します。

各epochは`Epoch`、`lr`、`train_loss`、`test_loss`、`acc`をRichで色分けします。文字としての書式は次のとおりです。

```text
Epoch [  1/20]  lr=1.00e-03  train_loss=1.2345  test_loss=1.1234  acc=60.00%
```

best weightの更新は画面へ出力しません。

## 出力

実行ごとに`run.log_dir/YYYYMMDD_HHMMSS/`を作り、次を保存します。

- `config.json`: 実際に使用した設定
- `training_info.json`: device、PyTorch、モデル、MACs、入力の基本情報
- `metrics.jsonl`: epochごとのlossとaccuracy
- `curves.png`: train/test lossとtest accuracyの曲線。accuracyは赤線で、凡例にlatestとbestを表示
- `model_best.pth`: test accuracyが最良だったモデルの`state_dict`
- `model_final.pth`: 最終epochのモデルの`state_dict`

独自のloggerや`training.log`は作りません。量子化、Observer、resume checkpointなどは、必要になるまで含めない方針です。
