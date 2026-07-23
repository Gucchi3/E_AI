# E_AI

Embedded AIの研究に必要な範囲へ絞った、CIFAR-10用の小さなCNN学習コードです。FP32学習と基礎的な整数QATに対応しています。学習と評価の入口は`main.py`だけで、設定はJSONへまとめています。`tools/`内の確認用コードは各ファイルを直接実行します。

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

先頭と最後の層をINT8、中間層をFP4 E2M1にした混合精度QATモデルは、専用の設定で学習します。

```powershell
python main.py --config config_mixed_qat.json
```

`.pth`の中身を確認する場合は、確認用コードを直接実行します。

```powershell
python tools/print_pth.py log/20260719_120000/model_best.pth
```

保存時にBNをfoldした重みを整数化し、保存済み整数Tensorと一緒に確認する場合は`--integer`を付けます。

```powershell
python tools/print_pth.py log/20260719_120000/model_best.pth --integer
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
- `quantization.activation_range_momentum`: 活性rangeの移動平均。Q_ViTと同じ既定値は`0.95`
- `train.epochs`、`train.learning_rate`、`train.minimum_learning_rate`
- `train.weight_decay`、`train.label_smoothing`
- `train.mixup_alpha`、`train.cutmix_alpha`: `0`で個別に無効化
- `train.mixup_probability`: MixUpまたはCutMixを適用する確率
- `train.mixup_switch_probability`: 両方が有効な場合にCutMixを選ぶ確率

MixUpとCutMixはQ_ViTと同じバッチ単位で適用します。学習accuracyは混合比率に応じた期待正解率です。両方を無効にする場合は`mixup_alpha`と`cutmix_alpha`を`0`にします。

FP32モデルから保存したfold済み重みでQATを開始する場合は、`config.json`を次のように設定します。

```json
"model": {
  "name"        : "qat_cifar_cnn",
  "num_classes" : 10,
  "load_weight" : true,
  "weight_path" : "log/20260718_120000/model_best.pth"
}
```

raw `state_dict`、`model`キーを持つcheckpoint、`state_dict`キーを持つcheckpointを読み込めます。FP32モデルの保存時には、PyTorchの`fuse_conv_bn_eval()`でConvとBatchNormを融合し、BNキーを除いた重みを出力します。このfold済み重みは、同じConvキーを持つQATモデルへ読み込めます。

## QAT

`qat_cifar_cnn`は、FP32モデルの保存時にBNをfoldした重みを`QuantConv2d`へ読み込み、重みを出力チャンネル単位、ReLU後の活性をTensor単位でFake Quantizationします。QATモデル自身はBatchNormを持ちません。fold済みの重みを出力チャンネル単位の整数、biasを`input_scale * weight_scale`に対応するint32として量子化します。ConvとLinearはdequantize済みのFake Quantization Tensorで通常演算します。PULP式multiplierと右shiftへの変換は学習後に行い、QATのforwardには入れません。入力画像は保存形式と実機では0～255の`uint8`とし、PyTorch内では`ToTensor()`後の0～1へ`scale=1/255`を適用して同じ整数値を模擬します。

FP32モデルとQATモデルは平均プーリングを使用しません。空間方向は畳み込みで1×1まで変換し、Flatten後に線形層へ入力します。

活性scaleは学習中に`0.95 × previous + 0.05 × current`で更新し、評価時は固定します。scaleと整数biasは重みと同じ`state_dict`へ保存されます。最終Linearはクラスごとのaccumulator scaleを維持し、共通出力scaleへの変換は行いません。

量子化部品は`utils/quantization/`内で完結しており、整数とFP4 E2M1のFake Quantizationに対応します。`QuantConv2d`と`QuantLinear`は`quantizer`引数で両方式を切り替えます。実装済み機能と保留機能は`utils/quantization/README.md`へ一覧化しています。

`mixed_qat_cifar_cnn`では、最初の畳み込みをINT8入力×INT8重みで計算し、その出力をFP4へ量子化します。中間の畳み込みはFP4入力×FP4重みで計算します。最後の畳み込み出力をINT8へ量子化し、最後の全結合をINT8入力×INT8重みで計算します。biasは各層の入力scaleとweight scaleの積を使うsigned int32です。

学習率は`CosineAnnealingLR`により、`train.learning_rate`から`train.minimum_learning_rate`へ滑らかに低下します。epoch表示の`lr`には、そのepochで実際に使用した値が表示されます。

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
- `model_best.pth`: test accuracyが最良だったモデルの`state_dict`。FP32モデルはBNをConvへfoldして保存
- `model_final.pth`: 最終epochのモデルの`state_dict`。FP32モデルはBNをConvへfoldして保存

独自のloggerや`training.log`は作りません。分布可視化observer、resume checkpoint拡張などは、必要になるまで含めない方針です。
