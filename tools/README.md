# Tools

このフォルダのPythonコードは、`main.py`やE_AI内のmoduleを経由せず、リポジトリのルートから直接実行します。各ファイルはコマンドライン引数を受け取る`main()`と`if __name__ == "__main__":`を持つ単独実行CLIにします。

## .pthの表示

Tensorは形状、型、値を表示します。大きなTensorの値は通常省略されます。

```powershell
python tools/print_pth.py log/20260719_120000/model_best.pth
```

全要素を表示する場合は`--full`を付けます。

```powershell
python tools/print_pth.py log/20260719_120000/model_best.pth --full
```

量子化重みと保存済みの整数型Tensorだけを表示する場合は`--integer`を付けます。FP32モデルのConvとBatchNormは`.pth`保存時にfold済みであり、このコマンドは保存済みのConv重みを整数へ変換します。重みbit数と丸め方式は`.pth`と同じフォルダの`config.json`から読み込みます。

```powershell
python tools/print_pth.py log/20260719_120000/model_best.pth --integer
```

`config.json`がない場合や設定を変更する場合は明示できます。

```powershell
python tools/print_pth.py model_best.pth --integer --weight-bits 8 --rounding ties_away_from_zero
```

整数重みを含む全要素を表示する場合は併用できます。

```powershell
python tools/print_pth.py log/20260719_120000/model_best.pth --integer --full
```
