# 情報処理の応用B

## 講義情報

- 対象：立正大学データサイエンス学部
- 科目群：専門科目・専門基礎科目群
- 必選区分：選択

## 講義ノート

- 公開ページ: <https://tnarizuka.github.io/Applied_Information_Processing/>
- GitHub repository: <https://github.com/tnarizuka/Applied_Information_Processing>

本講義ノートは [Jupyter Book](https://jupyterbook.org/en/stable/intro.html) を用いて作成しています。

## 運用ドキュメント

- [リポジトリ概要](docs/repository-overview.md)
- [運用ガイド](docs/operation-guide.md)
- [保守チェックリスト](docs/maintenance-checklist.md)
- [トラブルシューティング](docs/troubleshooting.md)
- [公開用変更履歴](docs/public-changelog.md)

管理者用の作業メモは `private/` に置き、Git 管理および Jupyter Book の公開対象から除外します。

## 基本コマンド

```bash
pip install -r requirements.txt
jb build --all .
```

GitHub Pages への公開を含む通常更新は、内容確認後に次を実行します。

```bash
./build_push.sh "commit message"
```

## 参考書

- [総務省政策統括官政策基準部編集，高校からの統計・データサイエンス活用～上級編～，日本統計協会，2017．](https://www.soumu.go.jp/toukei_toukatsu/info/guide/stkankyo.htm)

- 竹村彰通・姫野哲人・高田聖治編，データサイエンス入門，学術図書出版社，2019．
