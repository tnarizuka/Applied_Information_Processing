# 運用ガイド

## 通常の更新手順

1. 本文・notebook・データ・図を編集する。
2. `jb build --all .` でビルドする。
3. `_build/html/index.html` をブラウザで確認する。
4. `git status --short` で差分を確認する。
5. 問題がなければ commit し，`main` に push する。
6. `ghp-import -n -p -f _build/html` で GitHub Pages を更新する。

通常は次のスクリプトで一連の処理を実行できます。

```bash
./build_push.sh "commit message"
```

## 公開前チェック

- ビルドが警告なしで完了する。
- 章ページ，図，数式，引用，参考文献が表示される。
- `.ipynb` と `.md` のダウンロードボタンが期待通り動く。
- `_build/` と `private/` が Git 管理に入っていない。
- 管理者用メモや内部事情を `docs/` に書いていない。

## GitHub Pages

公開ページは `gh-pages` ブランチから配信します。`main` ブランチにはソースのみを置き，HTML 生成物は Git 管理しません。

## 管理者用メモ

作業記録，公開メモ，復旧メモなど，一般公開しない情報は `private/` に置きます。公開できる手順や方針は，個人情報・内部事情・未公開予定を除いたうえで `docs/` に移します。
