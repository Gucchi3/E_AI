# E_AI リポジトリ設計

## 目的

E_AIは、Embedded AI研究向けの小さなCIFAR-10 CNNを、少ない依存関係と明確な責務で学習するプロジェクトです。現在はFP32、整数QAT、先頭・末尾INT8／中間FP4 E2M1の混合精度QATを扱います。QAT中はFake Quantizationを行い、PULP式parameterへの変換は学習後の責務とします。分布可視化observer、C配列export、resume checkpointは実装していません。

すべての実行は`main.py`から開始し、設定は`config.json`で管理します。新しい実行modeを追加するときだけ`utils/workflows.py`と`main.py`の`WORKFLOWS`を拡張します。

## 構成

```text
E_AI/
├── main.py                 # configを読み、workflowを選ぶ入口
├── config.json             # 学習設定
├── tools/                  # 単独実行する確認用CLI
├── model/
│   ├── __init__.py         # 明示的なmodel factory
│   ├── cnn.py              # TinyCifarCNN
│   └── qat_cnn.py          # TinyQATCNNとTinyMixedQATCNN
└── utils/
    ├── artifacts.py        # 実行結果、曲線、重みの保存
    ├── augmentation.py     # バッチ単位のMixUpとCutMix
    ├── config.py           # JSON読込、型変換、検証
    ├── data.py             # CIFAR-10 transformとDataLoader
    ├── display.py          # Q_ViT準拠のRich表示
    ├── engine.py           # 1 epochのtrainとevaluate
    ├── profiling.py        # MACsの計測
    ├── quantization/       # 単独でコピーできる量子化部品
    │   ├── batch_norm.py   # BatchNorm fold
    │   ├── fp4.py          # FP4 E2M1 Fake Quantization
    │   ├── integer.py      # 整数Fake Quantization
    │   ├── layers.py       # 量子化ConvとLinear
    │   ├── requantization.py # PULP式再量子化
    │   └── rounding.py     # 丸め方式
    ├── runtime.py          # seedとdeviceの選択
    ├── weights.py          # 学習済み重みの読込
    └── workflows.py        # 学習全体の組み立て
```

`utils/quantization/`はPyTorchと同じフォルダ内のmodule以外をimportしません。`model/qat_cnn.py`だけが公開APIの`utils.quantization`を利用します。`utils/engine.py`は任意の`nn.Module`を受け取り、モデル名や保存先を知りません。`utils/workflows.py`だけがmodel、data、engine、artifactを組み合わせます。

`tools/`内のPythonコードはE_AI内のmoduleへ依存させず、各ファイルを直接実行できるCLIにします。学習・評価workflowから呼び出しません。

## 表示の契約

- 学習設定は`rich.Table`の「Training Configuration」で表示する。
- 表示項目名と並びは、実装済みの設定に限ってQ_ViTへ合わせる。
- epoch結果はRichで項目ごとに色分けし、文字の並びはQ_ViTの簡潔な書式に合わせる。
- MACsは`thop.profile`で計測する。計測エラーは`N/A`へ変換せず、そのまま表示する。
- `model.load_weight=true`の場合は、optimizerを作る前に`model.weight_path`のstate_dictを読み込む。
- best weightの更新や最終weightの保存について独自メッセージを追加しない。
- 標準`logging`を学習表示へ持ち込まない。

## コードスタイル

- 関数定義と関数呼び出しの引数は1行に書く。
- 関連する代入文が連続する場合は`=`の位置を揃える。
- 関数の間は2行、クラスの間は3行空ける。クラス内のmethodにも2行の空行を入れる。
- docstringとコメントは短い日本語で書く。
- この方針に合わせ、Ruffの行長上限は320文字とする。
- 機能を追加するときは、現在の責務へ収まるかを先に確認する。

## 出力

`utils/artifacts.py`は`log/YYYYMMDD_HHMMSS/`へ次を保存します。

| ファイル | 内容 |
| --- | --- |
| `config.json` | 検証済みの実行設定 |
| `training_info.json` | device、PyTorch、モデル、MACs、入力の基本情報 |
| `metrics.jsonl` | epochごとのlr、loss、accuracy |
| `curves.png` | latestとbestを凡例に含むlossとaccuracyの曲線 |
| `model_best.pth` | test accuracyが最良だった重みと量子化bufferを含む`state_dict` |
| `model_final.pth` | 最終epochの重みと量子化bufferを含む`state_dict` |

QATモデルでは量子化scale、整数bias、bias scaleを同じ`state_dict`へ保存します。FP4QuantizerはEMA計算途中のrangeを保存せず、最終scaleだけを保存します。最終Linearの出力scaleは共通化しません。

`training.log`は作りません。モデルファイルはoptimizerやepochを含まないため、resume checkpointではありません。
