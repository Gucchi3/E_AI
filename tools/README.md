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
