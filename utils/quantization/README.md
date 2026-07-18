# Quantization

このフォルダはPyTorch以外のプロジェクト内依存を持ちません。`quantization/`をコピーし、コピー先のpackageからimportすれば使用できます。

## 整数量子化

`IntegerQuantizer`は2、4、8、16 bitの符号付き・符号なしFake Quantizationに対応します。既定の丸めは中間値を絶対値が大きい側へ丸める`ties_away_from_zero`です。

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

重みのscaleをチャネル単位にする場合は`channel_axis=0`を指定します。`QuantConv2d`と`QuantLinear`ではこの設定を使用しています。

## 入力画像

画像入力は`uint8`の0～255を基準にします。PyTorchの学習では`ToTensor()`が画像を0～1へ変換するため、8 bit入力のscaleを`1 / 255`に固定して同じ整数値を再現します。

```python
input_quantizer = IntegerQuantizer(bit_width=8, signed=False, fixed_scale=1.0 / 255.0)
```

## 現在含めていない機能

- Observerとcalibration
- scaleのfreezeとunfreeze
- BatchNormのfold
- 整数演算だけで行うrequantization
- FP4Quantizer

これらは基礎的な整数QuantizerとQATモデルを検証した後に追加します。
