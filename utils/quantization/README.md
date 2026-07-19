# Quantization

このフォルダはPyTorch以外のプロジェクト内依存を持ちません。`quantization/`をコピーし、コピー先のpackageからimportすれば使用できます。

## 整数量子化

`IntegerQuantizer`は2、4、8、16 bitの符号付き・符号なしFake Quantizationに対応します。既定の丸めは中間値を絶対値が大きい側へ丸める`ties_away_from_zero`です。量子化器が`state_dict`へ保存するbufferは`scale`だけです。

```python
import torch

from utils.quantization import IntegerQuantizer


quantizer      = IntegerQuantizer(bit_width=8, signed=True)
fake_quantized = quantizer(torch.tensor([-1.0, -0.5, 0.5, 1.0]))
integer_result = quantizer.quantize(torch.tensor([-1.0, -0.5, 0.5, 1.0]))

print(fake_quantized)
print(integer_result.values)
print(integer_result.scale)
```

`forward()`はSTEを使うためQATで勾配を計算できます。`quantize()`は推論実装の確認に使う整数値、scale、zero pointを返します。現在のzero pointは常に0です。

重みのscaleをチャンネル単位にする場合は`channel_axis=0`と`channel_size`を指定します。`QuantConv2d`、`QuantBNConv2d`、`QuantLinear`では出力数を`channel_size`として使用しています。重みscaleは現在の重みからforwardごとに計算し、最新値をbufferへ保存します。

## FP4量子化

`FP4Quantizer`はE2M1形式の`0`、`±0.5`、`±1`、`±1.5`、`±2`、`±3`、`±4`、`±6`へFake Quantizationします。4bit codeは`quantize()`、復元値は`dequantize()`で取得できます。活性で`range_momentum`を指定した場合はscale自体をEMA更新し、`state_dict`には最終scaleだけを保存します。

```python
from utils.quantization import FP4Quantizer, QuantConv2d


activation_quantizer = FP4Quantizer(range_momentum=0.95)
convolution          = QuantConv2d(3, 16, kernel_size=3, quantizer="fp4", weight_bits=4)
```

整数とFP4は同じ`QuantConv2d`、`QuantBNConv2d`、`QuantLinear`を使用し、`quantizer="integer"`または`quantizer="fp4"`で重みQuantizerを選びます。biasはどちらも`input_scale * weight_scale`をscaleとするsigned int32格子へFake Quantizationします。

## 活性scale

活性Quantizerではscaleを移動平均で更新します。既定の`range_momentum=0.95`では、現在のscaleを5%、以前のscaleを95%として使用します。

```text
scale = 0.95 × previous_scale + 0.05 × current_scale
```

学習modeではscaleを更新し、評価modeでは保存済みのscaleを使用します。評価中のscaleはtest batchの内容によって変化しません。学習前でscaleが未初期化のモデルを評価すると例外を出します。

```python
activation_quantizer = IntegerQuantizer(bit_width=8, signed=False, range_momentum=0.95)

activation_quantizer.train()
activation_quantizer(train_activation)

activation_quantizer.eval()
test_activation = activation_quantizer(test_activation)
```

## 現在のQATフロー

1. 入力は`uint8`に対応する固定scale `1/255`でFake Quantizationする。
2. 重みはforward時の現在値からチャンネル別scaleを計算する。
3. 活性は現在のTensorから求めたscaleを学習中にEMAで更新する。
4. ConvとLinearはdequantize済みのFake Quantization Tensorで通常演算する。
5. Convブロック後の活性を`IntegerQuantizer`でFake Quantizationする。
6. 評価時は活性scaleを更新せず、学習中に確定したscaleを使用する。
7. best/finalモデル保存時は重みと量子化bufferを同じ`state_dict`へ保存する。

`QuantBNConv2d`は学習中にrunning統計を更新しながら、running統計でfoldした重みのFake Quantizationを行います。評価時はBatchNormを重みとbiasへ完全にfoldするため、推論グラフにBatchNorm演算は残りません。fold後のbias scaleは`input_scale * weight_scale`で、int32としてFake Quantizationします。`weight_quantizer.scale`、`bias_integer`、`bias_scale`はstate_dictへ保存されます。

`QuantConv2d`と`QuantLinear`はbiasを持つ場合に入力scaleを必須とし、入力scaleと出力単位のweight scaleから`bias_scale`を計算します。QATの勾配更新に必要なFP32 master biasとは別に、実機で使用する`bias_integer`と`bias_scale`をbufferとしてstate_dictへ保存します。

学習時のFP32 master weight、BatchNormのbatch統計、損失計算は勾配更新のために残します。PyTorchの`conv2d`と`linear`へ渡すのは、選択した形式へ量子化してからdequantizeした浮動小数点Tensorです。量子化codeを直接ConvやLinearへ渡しません。最終Linearはクラス別scaleのまま実数logitを返し、共通scale化は実機argmaxの仕様を決めるまで行いません。

## 学習後の固定小数点変換

`FixedPointRequantizer`と`fixed_point_parameters()`は、学習後に入力scaleと出力scaleの比率をPULP式のmultiplierと右shiftへ変換するための部品です。現在のQATモデルのforwardには接続しません。

```text
real_multiplier = input_scale / output_scale
real_multiplier ≒ multiplier / 2^shift
output_integer   = clip((input_integer × multiplier) >> shift)
```

QAT完了後にBN fold、整数重み・bias生成、accumulator上限計算を行ってから使用します。FP4をCV32E40P向け整数係数へ展開する処理も、この学習後変換の責務です。

MAC計測はモデルのコピーで行い、本物のモデルの活性scaleとBatchNorm統計を変更しません。

## 入力画像

画像入力は`uint8`の0～255を基準にします。PyTorchの学習では`ToTensor()`が画像を0～1へ変換するため、8 bit入力のscaleを`1 / 255`に固定して同じ整数値を再現します。

```python
input_quantizer = IntegerQuantizer(bit_width=8, signed=False, fixed_scale=1.0 / 255.0)
```

## 機能範囲

| 機能 | 状態 | 今回の判断 |
| --- | --- | --- |
| INT2/4/8/16 Fake Quantization | 実装済み | QATの基礎として必要 |
| FP4 E2M1 Fake Quantization | 実装済み | 3種類の共通丸めとscale EMAに対応 |
| ties-away-from-zero丸め | 実装済み | CV32E40P向けの丸め一致に必要 |
| 重みのチャンネル別scale | 実装済み | ConvとLinearの重み量子化に必要 |
| 活性scaleのEMA | 実装済み | 安定したQATに必要 |
| 評価時の活性scale固定 | 実装済み | batch非依存の評価に必要 |
| scaleのstate_dict保存 | 実装済み | FP4は途中のrangeを保存しない |
| uint8入力の固定scale | 実装済み | エッジ入力との対応に必要 |
| MAC計測時の状態保護 | 実装済み | ダミー入力によるscale汚染を防ぐため必要 |
| 旧checkpointの重み読込 | 実装済み | 量子化bufferがない旧重みからQATを開始するため必要 |
| 分布可視化observer | 保留 | 現在のQAT学習には不要 |
| 手動calibration | 保留 | 学習済みEMAを検証してから追加 |
| BatchNorm fold | 実装済み | fold後の重みをper-channel量子化 |
| fold後biasのint32量子化 | 実装済み | `input_scale * weight_scale`を使用 |
| Linear biasのint32量子化 | 実装済み | 整数値とscaleをstate_dictへ保存 |
| PULP式整数requantization | 変換部品のみ | QAT後の変換で使用する |
| Linear出力scaleの共通化 | 保留 | 実機argmaxの比較方式を決めてから実装 |
| 重み・scaleのC配列export | 保留 | 推論レイアウト確定後に追加 |

今回は、学習・評価・checkpoint保存を正しく行うために必要な範囲だけを実装しています。
