# E_AI

Embedded AIの研究に必要な範囲へ絞った、CIFAR-10用の小さなCNN学習コードです。FP32、INT8、INT4、FP4 E2M1、UFP4 E2M2、INT8/FP4混合精度の学習に対応しています。学習と評価の入口は`main.py`だけで、設定はJSONへまとめています。`tools/`内の確認用コードは各ファイルを直接実行します。

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

`python main.py`は既定で`config/basic_cnn/cnn.json`を使用します。各モデルの設定は次のコマンドで指定できます。

```powershell
python main.py --config config/basic_cnn/cnn.json
python main.py --config config/basic_cnn/int8_cnn.json
python main.py --config config/basic_cnn/int4_cnn.json
python main.py --config config/basic_cnn/fp4_cnn.json
python main.py --config config/basic_cnn/mixed_cnn.json
python main.py --config config/basic_cnn/test_cnn.json
python main.py --config config/basic_cnn/ufp4_test_cnn.json
python main.py --config config/ResNet18/resnet18_fp32.json
python main.py --config config/ResNet18/resnet18_int8.json
python main.py --config config/ResNet18/resnet18_int4.json
python main.py --config config/ResNet18/resnet18_fp4.json
python main.py --config config/ResNet18/resnet18_ufp4.json
python main.py --config config/MobileNet_v2/mobilenet_v2_fp32.json
python main.py --config config/MobileNet_v2/mobilenet_v2_int8.json
python main.py --config config/MobileNet_v2/mobilenet_v2_int4.json
python main.py --config config/MobileNet_v2/mobilenet_v2_fp4.json
python main.py --config config/MobileNet_v2/mobilenet_v2_ufp4.json
```

`resnet18_fp32`は3×3・stride 1のstemとmax poolingなしで32×32入力へ適応したResNet-18、`mobilenet_v2_fp32`はstemのstrideだけを1へ変更したCIFAR-10向けMobileNetV2です。既存のCNN系モデルは`model/basic_cnn/`、ResNet-18は`model/ResNet18/`、MobileNetV2は`model/MobileNet_v2/`に整理しています。

モデル名と実装ファイル、設定ファイルは`cnn`、`int8_cnn`、`int4_cnn`、`fp4_cnn`、`mixed_cnn`、`test_cnn`、`ufp4_test_cnn`で統一しています。

`.pth`の中身を確認する場合は、確認用コードを直接実行します。

```powershell
python tools/print_pth.py log/20260723_220608/model_best.pth
```

保存時にBNをfoldした重みを整数化し、保存済み整数Tensorと一緒に確認する場合は`--integer`を付けます。

```powershell
python tools/print_pth.py log/20260723_220608/model_best.pth --integer
```

主な設定は次のとおりです。

- `run.device`: `auto`、`cpu`、`cuda`
- `run.log_dir`: 実行結果を保存する親ディレクトリ
- `data.image_size`: `32`または`256`
- `data.normalization`: CIFAR-10統計を使う`cifar10`、または`[0, 1]`の`zero_one`
- `data.batch_size`、`data.num_workers`
- `model.load_weight`: 学習済み重みを読み込む場合は`true`
- `model.weight_path`: 読み込む`.pth`ファイルのパス
- `quantization.weight_bits`: モデルが使用する重みbit数。INT8は`8`、INT4・FP4・Mixedは`4`
- `quantization.activation_bits`: モデルが使用する活性bit数。INT8は`8`、INT4・FP4・Mixedは`4`
- `quantization.input_bits`: `uint8`入力に合わせて現在は`8`
- `quantization.rounding`: `ties_away_from_zero`、`ties_to_positive`、`ties_to_even`
- `quantization.activation_range_momentum`: 活性rangeの移動平均。Q_ViTと同じ既定値は`0.95`
- `train.epochs`、`train.learning_rate`、`train.minimum_learning_rate`
- `train.weight_decay`、`train.label_smoothing`
- `train.mixup_alpha`、`train.cutmix_alpha`: `0`で個別に無効化
- `train.mixup_probability`: MixUpまたはCutMixを適用する確率
- `train.mixup_switch_probability`: 両方が有効な場合にCutMixを選ぶ確率

MixUpとCutMixはQ_ViTと同じバッチ単位で適用します。学習accuracyは混合比率に応じた期待正解率です。両方を無効にする場合は`mixup_alpha`と`cutmix_alpha`を`0`にします。

FP32モデルから保存したfold済み重みで量子化学習を開始する場合は、対象モデルの設定を次のようにします。付属の量子化モデル用設定はすべてこの読込先を指定しています。

```json
"model": {
  "name"        : "int8_cnn",
  "num_classes" : 10,
  "load_weight" : true,
  "weight_path" : "log/20260723_220608/model_best.pth"
}
```

raw `state_dict`、`model`キーを持つcheckpoint、`state_dict`キーを持つcheckpointを読み込めます。FP32モデルの保存時には、PyTorchの`fuse_conv_bn_eval()`でConvとBatchNormを融合し、BNキーを除いた重みを出力します。このfold済み重みは、同じConvキーを持つQATモデルへ読み込めます。

## QAT

量子化モデルは、FP32モデルの保存時にBNをfoldした重みを`QuantConv2d`へ読み込みます。量子化モデル自身はBatchNormを持ちません。fold済みの重みを出力チャンネル単位、活性をTensor単位でFake Quantizationし、biasは`input_scale * weight_scale`に対応するint32として量子化します。ConvとLinearはdequantize済みのFake Quantization Tensorで通常演算します。PULP式multiplierと右shiftへの変換は学習後に行い、QATのforwardには入れません。入力画像は保存形式と実機では0～255の`uint8`とし、PyTorch内では`ToTensor()`後の0～1へ`scale=1/255`を適用して同じ整数値を模擬します。

- `int8_cnn`: 重みと活性をINT8で量子化
- `int4_cnn`: 先頭と末尾をINT8、中間の重みと活性をINT4で量子化
- `fp4_cnn`: 先頭と末尾をINT8、中間の重みと活性をFP4 E2M1で量子化
- `mixed_cnn`: 先頭と末尾をINT8、中間をFP4 E2M1で量子化
- `test_cnn`: 先頭と末尾をINT8、中間をUINT4活性×FP4 E2M1重みで量子化
- `ufp4_test_cnn`: 先頭と末尾をINT8、中間をUFP4 E2M2活性×FP4 E2M1重みで量子化

BasicViTは、`basic_vit_fp32`の後に`basic_vit_int8`、`basic_vit_int4`、`basic_vit_fp4`、`basic_vit_ufp4`の順で基本モデルを並べ、追加実験を`basic_vit_test1`～`basic_vit_test7`として管理します。

| モデル名 | 本体 | Attention | 残差 |
| --- | --- | --- | --- |
| `basic_vit_test1` | FP4重み、ReLU後UFP4 | FP4 | 独立scale INT8 |
| `basic_vit_test2` | FP4重み、ReLU後UFP4 | INT8 | 共有scale INT4 |
| `basic_vit_test3` | FP4重み、ReLU後UFP4 | INT8 | 独立scale INT8 |
| `basic_vit_test4` | FP4重み、ReLU後UFP4 | FP4 | 独立scale INT4 |
| `basic_vit_test5` | FP4重み、ReLU後UFP4 | FP4 | 共有scale INT8 |
| `basic_vit_test6` | INT4重み、signed INT4活性 | signed INT4 | 共有scale INT8 |
| `basic_vit_test7` | INT4重み、ReLU後UINT4 | signed INT4 | 共有scale INT8 |

すべてのConvは3×3です。平均プーリングは使用せず、32×32入力をstride 2のConvで`32→16→8→4`へ縮小し、`64×4×4`をFlattenして線形層へ入力します。

活性scaleは学習中に`0.95 × previous + 0.05 × current`で更新し、評価時は固定します。scaleと整数biasは重みと同じ`state_dict`へ保存されます。最終Linearはクラスごとのaccumulator scaleを維持し、共通出力scaleへの変換は行いません。

量子化部品は`utils/quantization/`内で完結しており、整数、FP4 E2M1、UFP4 E2M2のFake Quantizationに対応します。`QuantConv2d`と`QuantLinear`の重みは`quantizer`引数で整数またはFP4を切り替え、UFP4はReLU後の活性に使用します。実装済み機能と保留機能は`utils/quantization/README.md`へ一覧化しています。

`mixed_cnn`では、最初の畳み込みをINT8入力×INT8重みで計算し、その出力をFP4へ量子化します。中間の畳み込みはFP4入力×FP4重みで計算します。最後の畳み込み出力をINT8へ量子化し、最後の全結合をINT8入力×INT8重みで計算します。biasは各層の入力scaleとweight scaleの積を使うsigned int32です。

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
