# トラブルシューティング

## ビルドが失敗する

- `pip install -r requirements.txt` を実行する。
- notebook の JSON が壊れていないか確認する。
- `_toc.yml` に存在しないファイルを指定していないか確認する。
- 画像やデータファイルのパスが，参照元 notebook から見て正しいか確認する。

## 日本語フォントが表示されない

本文では `matplotlib-fontja` を推奨します。

```python
import matplotlib.pyplot as plt
import matplotlib_fontja
```

環境固有のフォントを直接指定する場合は，Windows と macOS で利用できるフォント名が異なる点に注意します。

## ダウンロードボタンで内容が表示される

`_static/download.js` を読み込み，Jupyter Book の source / notebook download link に `download` 属性を付けています。ビルド後の HTML に `_static/download.js` が含まれているか確認します。

## GitHub Pages が更新されない

- `main` に最新ソースが push されているか確認する。
- `ghp-import -n -p -f _build/html` が成功しているか確認する。
- GitHub Pages の公開元が `gh-pages` ブランチになっているか確認する。
- ブラウザや GitHub Pages のキャッシュにより反映に時間がかかる場合があります。
