# E_AI QAT引き継ぎ書

作成日: 2026-07-19

## 2026-07-19 修正結果

この文書に記録した誤実装は項目ごとに修正した。以下のGit状態と差分概要は修正前の記録として残す。

- `QuantConv2d`を汎用部品として復元した。
- `QuantBNConv2d`と`QuantLinear`は、dequantize済みFake Quantization Tensorで通常演算する形へ戻した。
- 最終Linearの共通scale化を削除した。
- Convブロック後の固定小数点再量子化だけを独立処理として残した。
- Q31 `multiplier`と右`shift`をsigned int32 bufferへ変更した。
- ConvとLinearの整数bias、bias scaleをstate_dictへ保存する。
- 新しいobserver、FP4、C配列export、overflow模擬は追加していない。

学習用scaleと活性rangeはFP32で保持し、CV32E40Pへ渡す再量子化parameterをint32で保存する。最終Linearの実機argmax方式と、CV32E40P上の積を`mul`、`mulh`などで処理する命令列は未決である。

## 最重要事項

以下は修正前の注意事項である。

直前の実装では、Fake QuantizationによるQATと整数推論エミュレーションを一部混同している。ユーザーから次の修正指示が出ている。

1. QAT中の畳み込みとLinearは、量子化してからdequantizeした浮動小数点Tensorで通常の`conv2d`、`linear`を行う。整数コードを直接演算させる必要はない。
2. 現在のモデルで未使用でも、汎用部品として`QuantConv2d`は残す。
3. 最終Linearの出力scale共通化は、自動的に正しい処理とは限らない。必要性を再検討する。
4. CV32E40Pへ渡す量子化scale表現と再量子化パラメータは、すべてint32に収まる形にする。
5. 現在の未コミット差分には正しい修正と誤った修正が混在する。`git reset`などで一括破棄せず、差分を項目ごとに確認する。

## E_AIの目的

E_AIは、Q_ViTと同じ使い勝手を保ちながら、Embedded AI研究に必要な範囲へ機能を絞った、単純で保守しやすい学習コードにする。

最終目的は次のとおり。

1. QAT学習を行う。
2. 学習済み重み、bias、活性range、scaleを保存する。
3. 保存値をCV32E40P向けの整数表現へ変換する。
4. CV32E40P上で整数推論を実行する。

QAT学習コードとCV32E40P用整数推論コードは、責務を混同しないこと。QATではFake Quantizationを使用し、実機用整数コードと固定小数点パラメータは保存・変換用の値として扱う。

## ユーザーが確定した方針

- 入力画像は0～255の`uint8`とする。
- PyTorch内では`ToTensor()`後の0～1に固定scale `1/255`を適用する。
- 入力zero pointは0とする。
- ReLU後の活性は非負なので、zero point 0のunsigned量子化を使用する。
- 整数量子化は2、4、8、16 bitを同じQuantizerで扱う。
- FP4は将来`FP4Quantizer`として別途実装する。
- 丸め関数名はCV32E40P専用名にせず、`ties_away_from_zero`など丸め方式の名前にする。
- 現在の既定丸めは`ties_away_from_zero`。
- overflowの厳密な模擬は現段階では行わない。
- 活性rangeはEMAで更新する。既定値0.95では、以前のrangeを0.95、現在のrangeを0.05使用する。
- 評価時は活性rangeとscaleを固定する。
- scale、range、量子化用bufferは`state_dict`へ保存する。
- BatchNormは推論時に畳み込み重みとbiasへfoldする。
- モデルでは平均プーリングを使用しない。畳み込みとLinearを中心に構成する。
- observer、freeze、unfreeze、FP4、C配列exportは、現在の最低限実装には含めない。
- `utils/quantization/`は、このフォルダだけコピーして他のプロジェクトでも使える構成にする。E_AI固有コードへ依存させない。

## コードスタイル

- 関数の引数は必ず1行で書く。
- 代入文は同じ処理ブロック内で`=`の位置を揃える。
- 関数間は空行を2行入れる。
- クラス間は空行を3行入れる。
- docstringとコメントは日本語で書く。
- AIが生成したと分かるような説明調の文章をコード内へ増やさない。
- 不要なloggerや独自ログ文章を追加しない。
- Rich表示はQ_ViTの使い勝手を基本とする。

## 直前までに実装済みの機能

以下は直前のQ31修正より前から実装されていた機能を含む。

- Richによる設定・学習状況表示
- MAC数計測
- 学習済み重み読込設定
- CosineAnnealingLR
- label smoothing
- MixUp、CutMix
- 活性rangeのEMA
- 評価時のrange固定
- scale、rangeの`state_dict`保存
- BatchNorm fold対応QAT CNN
- 平均プーリングを使わないCNN
- Linear biasをint32格子へFake Quantizationする修正

これらを一括で消さないこと。

## 現在のGit状態

基準コミットは`592aaeb`。

現在は未コミット・未ステージの差分がある。

```text
 M README.md
 M docs/repository_design.md
 M model/qat_cnn.py
 M utils/profiling.py
 M utils/quantization/README.md
 M utils/quantization/__init__.py
 M utils/quantization/integer.py
 M utils/quantization/layers.py
 M utils/weights.py
?? utils/quantization/requantization.py
```

直前の差分概要は次のとおり。

- `utils/quantization/requantization.py`
  - Q31 multiplierと右shiftによる再量子化を新規実装した。
  - 現在の`multiplier`はint64 bufferであり、ユーザーの最新方針と一致しない。
  - 内部の入力整数と積もint64で処理している。
- `utils/quantization/layers.py`
  - `QuantConv2d`を削除してしまった。復元が必要。
  - `QuantBNConv2d`の学習時・評価時畳み込みを、整数コードを復元して演算する形へ変更してしまった。Fake Quantization方式へ戻す必要がある。
  - Convのfold後biasをint32格子へ丸め、`bias_integer`を保存する処理を追加した。この考え方は維持候補。
  - `QuantLinear`のweightとbiasを整数コードへ戻して`F.linear`する処理を追加した。dequantize済みFake Quantization Tensorで演算する形へ直す。
  - 最終出力を共通int32 scaleへ再量子化する処理を追加した。必要性を再検討する。
- `model/qat_cnn.py`
  - 各Conv block後に`FixedPointRequantizer`を追加した。
  - 再量子化モジュール自体は必要だが、Fake Quantization経路との接続方法を再設計する。
- `utils/quantization/integer.py`
  - range更新とscale取得だけを行う`scale_for()`を追加した。
  - 再量子化モジュールからscaleだけ取得する用途であり、維持可能。
- `utils/profiling.py`
  - `QuantBNConv2d`がtupleを返す変更へ対応した。
  - `QuantConv2d`削除に合わせた処理も消しているため、復元時にMAC計測も戻す。
- `utils/weights.py`
  - 新しいbufferを持たない旧checkpointの読込を許可した。
- README類
  - 直前実装を完成扱いした説明が含まれる。コードの再修正後に内容を合わせる。

## Fake Quantizationの正しい責務

QATの基本経路は次の形にする。

```text
x_fake = clamp(round(x / sx)) * sx
w_fake = clamp(round(w / sw)) * sw
b_fake = clamp(round(b / sb)) * sb

y = conv2d(x_fake, w_fake, b_fake)
```

`x_fake`、`w_fake`、`b_fake`は整数格子上にある浮動小数点Tensorである。PyTorchの畳み込みやLinearにはこれらを渡す。QATのforward中に整数コードへ戻してから`conv2d`や`linear`を呼ぶ必要はない。

整数コードは、次の用途で取得する。

- checkpoint確認
- export
- CV32E40P用配列生成
- 固定小数点再量子化パラメータの検証
- 必要な場合だけ行う整数推論エミュレーション

QAT本体と整数推論エミュレータを同じレイヤー内部へ混在させないこと。

## 再量子化で維持すべき考え方

ConvまたはLinearのaccumulator scaleは、通常は次のとおり。

```text
accumulator_scale[channel] = input_scale * weight_scale[channel]
real_multiplier[channel]   = accumulator_scale[channel] / output_scale
```

実機では次のような固定小数点演算へ変換する。

```text
output_integer = round(accumulator_integer * multiplier / 2^shift)
```

ただし、次を修正・再検討する必要がある。

- `multiplier`はsigned int32に収める。
- `shift`もint32で保存・受け渡しする。
- `multiplier == 2^31`にならない正規化を行う。
- accumulatorとmultiplierの積をCV32E40Pでどう計算するかを、`mul`、`mulh`など実際の命令列に合わせて別途決める。
- QAT forwardではdequantize済みTensorによるConv・Linearを維持し、固定小数点誤差を入れる場所だけを再量子化モジュールに限定する。
- 既定丸め`ties_away_from_zero`を整数右shiftでも同じように適用する。

## 「scaleをすべてint32にする」の未決事項

ユーザーの最新要求は「CV32E40Pで扱えるように、量子化スケールの計算は、すべてint32に収まる範囲でやり取りする」である。

次のチャットでは、実装前に対象範囲を確認すること。

1. 学習中のobserver・EMAが持つ実数scaleはFP32のまま許可し、export時だけ`int32 multiplier + int32 shift`へ変換するのか。
2. checkpointにも実数scaleを残さず、固定小数点表現だけを保存したいのか。
3. `input_scale`、`weight_scale`、`output_scale`そのものを、どのQ形式でint32化するのか。

一般的なQATでは学習用scaleはFP32で保持し、実機へ渡す再量子化係数を`int32 multiplier + shift`へ変換する。ユーザーがcheckpoint内のscaleにもint32だけを求める場合は、Q形式を先に確定する必要がある。

## Linear出力scale共通化の問題

per-channel weight量子化では、Linearの各出力クラスが異なるaccumulator scaleを持つ。

QAT中にdequantize済みlogitをCross Entropyへ渡す場合は、各クラスのscaleを適用した実数logitを比較するため、共通scaleは不要。

共通scaleが必要になるのは、CV32E40P上で再scaleせず、生の整数logitだけを直接比較してargmaxしたい場合である。現在実装した「最大accumulator scaleを共通scaleにする」方式は比率を1以下にできるが、追加の丸め誤差と分解能低下が発生する。これは唯一の正解ではなく、ユーザーの最新指摘どおり再検討が必要。

次の修正では、原則としてQAT本体から共通scale処理を外す。実機argmaxの仕様を決めた後、exportまたは推論実装側で次のどれを採用するか決める。

- 共通scaleへ再量子化する。
- クラス別scaleを適用して比較する。
- cross multiplicationなど、除算を使わない比較方法を設計する。

## QuantConv2dの復元方針

`QuantConv2d`は現在のTinyQATCNNで未使用でも、`utils/quantization/`の汎用部品として残す。

最低限の責務は次のとおり。

- weightをper-channel Fake Quantizationする。
- biasがある場合は、入力scaleが渡されたときだけ`input_scale * weight_scale`でint32 Fake Quantizationできる設計を検討する。
- `F.conv2d`へ渡すのはdequantize済みFake Quantization Tensorとする。
- export用の整数weight、bias、scaleはbufferまたは明示的な取得関数から得られるようにする。
- E_AI固有モジュールへ依存させない。

## 次のチャットで最初に行う作業

1. このファイルと`git diff`を読む。
2. ユーザーへ「学習用FP32 scaleは許可し、実機へ渡す値だけint32にする理解でよいか」を確認する。
3. `QuantConv2d`を復元する。
4. `QuantBNConv2d`と`QuantLinear`を、dequantize済みFake Quantization Tensorで演算する形へ戻す。
5. Linear出力scale共通化をいったん外す。
6. Q31 multiplierをint32、shiftをint32にする。
7. 各Conv block後の固定小数点再量子化だけを、QATへ誤差を入れる独立処理として残す。
8. biasのint32 Fake Quantizationと保存は維持する。
9. train、eval、checkpoint保存・読込を実行確認する。
10. コード修正後にREADME類を更新する。

## 検証状況

直前のコードは以下のみ確認済み。

- `py_compile`通過
- `git diff --check`通過
- Q31係数の純粋な数値試験10万ケースで、直接丸めとの差が最大1 LSB

Codex付属PythonにはPyTorchが入っていなかったため、実モデルのforward、backward、1 epoch学習は未実行。次のチャットでは、ユーザーのPyTorch環境または適切な仮想環境を特定して実行確認すること。

## 実装範囲を増やさないこと

現在は最低限のQATを正しくする段階である。次の機能は、ユーザーから明示的な指示が出るまで追加しない。

- observer追加
- freeze、unfreeze
- FP4
- C配列export
- overflowエミュレーション
- resume checkpoint拡張
- 新しいlogger
- 独自の学習メッセージ
- モデル構造の拡張
