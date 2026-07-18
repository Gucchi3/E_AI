# Quantization

このフォルダはPyTorch以外のプロジェクト内依存を持ちません。`quantization/`をコピーし、コピー先のpackageからimportすれば使用できます。

## 整数量子化

`IntegerQuantizer`は2、4、8、16 bitの符号付き・符号なしFake Quantizationに対応します。既定の丸めは中間値を絶対値が大きい側へ丸める`ties_away_from_zero`です。`scale`、`running_min`、`running_max`、`range_initialized`はbufferとして`state_dict`へ保存されます。

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

## 活性range

活性QuantizerではQ_ViTと同じ移動平均を使用します。既定の`range_momentum=0.95`では、現在のrangeを5%、以前のrangeを95%として更新します。

```text
running_range = 0.95 × previous_range + 0.05 × current_range
```

学習modeではrunning rangeを更新し、評価modeでは保存済みのrangeとscaleを使用します。評価中のscaleはtest batchの内容によって変化しません。学習前でrangeが未初期化のモデルを評価すると例外を出します。

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
3. 活性は学習中のmin/maxをEMAで更新し、そのrangeからscaleを計算する。
4. ConvとLinearはdequantize済みのFake Quantization Tensorで通常演算する。
5. Convブロック後だけ、Q31 multiplierと右shiftによる再量子化誤差を加える。
6. 評価時は活性rangeを更新せず、学習中に確定したscaleを使用する。
7. best/finalモデル保存時は重みとすべての量子化bufferを同じ`state_dict`へ保存する。
8. checkpoint読込時はscaleとrunning rangeも復元し、QATを継続できる。

`QuantBNConv2d`は学習中にrunning統計を更新しながら、running統計でfoldした重みのFake Quantizationを行います。評価時はBatchNormを重みとbiasへ完全にfoldするため、推論グラフにBatchNorm演算は残りません。fold後のbias scaleは`input_scale * weight_scale`で、int32としてFake Quantizationします。`weight_quantizer.scale`、`bias_integer`、`bias_scale`はstate_dictへ保存されます。

`QuantConv2d`と`QuantLinear`は入力scaleを受け取った場合だけ、入力scaleと出力単位のweight scaleから`bias_scale`を計算し、biasをint32格子へFake Quantizationします。QATの勾配更新に必要なFP32 master biasとは別に、実機で使用する`bias_integer`と`bias_scale`をbufferとしてstate_dictへ保存します。

学習時のFP32 master weight、BatchNormのbatch統計、損失計算は勾配更新のために残します。PyTorchの`conv2d`と`linear`へ渡すのは、整数格子へ丸めてからdequantizeした浮動小数点Tensorです。整数コードを直接ConvやLinearへ渡しません。最終Linearはクラス別scaleのまま実数logitを返し、共通scale化は実機argmaxの仕様を決めるまで行いません。

## 固定小数点再量子化

`FixedPointRequantizer`は、入力scaleと出力scaleの比率をsigned int32のQ31 multiplierとint32右shiftへ変換します。

```text
real_multiplier = input_scale / output_scale
real_multiplier ≒ multiplier / 2^shift
output_integer   = round((input_integer × multiplier) / 2^shift)
```

forwardでは通常のConvやLinearとは独立してQ31再量子化誤差を加えます。各出力チャンネルの`multiplier`と`shift`はint32 bufferとしてstate_dictへ保存されます。accumulator overflowやCV32E40P上の命令列はまだ模擬しません。

MAC計測はモデルのコピーで行い、本物のモデルのrunning rangeとBatchNorm統計を変更しません。

## 入力画像

画像入力は`uint8`の0～255を基準にします。PyTorchの学習では`ToTensor()`が画像を0～1へ変換するため、8 bit入力のscaleを`1 / 255`に固定して同じ整数値を再現します。

```python
input_quantizer = IntegerQuantizer(bit_width=8, signed=False, fixed_scale=1.0 / 255.0)
```

## 機能範囲

| 機能 | 状態 | 今回の判断 |
| --- | --- | --- |
| INT2/4/8/16 Fake Quantization | 実装済み | QATの基礎として必要 |
| ties-away-from-zero丸め | 実装済み | CV32E40P向けの丸め一致に必要 |
| 重みのチャンネル別scale | 実装済み | ConvとLinearの重み量子化に必要 |
| 活性rangeのEMA | 実装済み | 安定したQATに必要 |
| 評価時の活性scale固定 | 実装済み | batch非依存の評価に必要 |
| scale/rangeのstate_dict保存 | 実装済み | checkpointと将来のexportに必要 |
| uint8入力の固定scale | 実装済み | エッジ入力との対応に必要 |
| MAC計測時の状態保護 | 実装済み | ダミー入力によるrange汚染を防ぐため必要 |
| 旧checkpointの重み読込 | 実装済み | 量子化bufferがない旧重みからQATを開始するため必要 |
| 分布可視化observer | 保留 | 現在のQAT学習には不要 |
| 手動calibration | 保留 | 学習済みEMAを検証してから追加 |
| BatchNorm fold | 実装済み | fold後の重みをper-channel量子化 |
| fold後biasのint32量子化 | 実装済み | `input_scale * weight_scale`を使用 |
| Linear biasのint32量子化 | 実装済み | 整数値とscaleをstate_dictへ保存 |
| Q31整数requantization | 実装済み | multiplierと右shiftをstate_dictへ保存 |
| Linear出力scaleの共通化 | 保留 | 実機argmaxの比較方式を決めてから実装 |
| 重み・scaleのC配列export | 保留 | 推論レイアウト確定後に追加 |
| FP4Quantizer | 保留 | 整数QATの検証後に追加 |

今回は、学習・評価・checkpoint保存を正しく行うために必要な範囲だけを実装しています。
