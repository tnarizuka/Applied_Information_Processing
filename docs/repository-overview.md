# リポジトリ概要

このリポジトリは，立正大学データサイエンス学部「情報処理の応用B」の講義ノートを Jupyter Book として管理するためのものです。

## 公開対象

- `index.md`
- `_toc.yml` に掲載された章ファイル
- 各章で参照される画像・CSV・Excel などの教材データ
- `reference.md`
- `README.md` と `docs/` 配下の公開用運用資料

## 非公開・管理対象外

- `_build/`: Jupyter Book の生成物
- `private/`: 管理者用メモ，作業記録，復旧メモ
- `.vscode/`: ローカルの VS Code 設定
- `*.pptx`: 編集用スライド素材

`private/` は Git 管理および Jupyter Book の公開対象から除外します。公開してよい内容は `docs/` に整理します。

## 主な設定

- `_config.yml`: Jupyter Book の表示，リポジトリリンク，HTML 追加設定
- `_toc.yml`: 公開する章構成
- `requirements.txt`: ビルドと notebook 実行に必要な Python パッケージ
- `build_push.sh`: ビルド，main への push，GitHub Pages 公開の補助スクリプト

## 依存関係の方針

本文で利用する主要パッケージは `requirements.txt` に記載します。日本語フォント設定は `japanize-matplotlib` ではなく，更新状況を踏まえて `matplotlib-fontja` を使う方針です。
