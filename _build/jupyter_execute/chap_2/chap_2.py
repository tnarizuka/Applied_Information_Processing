#!/usr/bin/env python
# coding: utf-8

# In[1]:


# モジュール・ライブラリのインポート（必ず最初に実行）
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 日本語フォントの設定（Mac:'Hiragino Sans', Windows:'MS Gothic'）
plt.rcParams['font.family'] = 'MS Gothic'

# 表記の設定
pd.set_option('display.precision', 3)   # 小数点以下の表示桁
pd.set_option('display.max_rows', 20)  # 表示する行数
pd.set_option('display.max_columns', 10)  # 表示する行数
get_ipython().run_line_magic('precision', '3')


# # プログラミング環境の構築

# ## Anacondaのインストール
# 
# 既に「プログラミング基礎」の授業内でAnacondaをインストールしているはずなので，以下ではインストールの概要だけ述べる．詳細は[Python.jp](https://www.python.jp/install/anaconda/windows/install.html)や[Let'sプログラミング](https://www.javadrive.jp/python/install/index5.html)などが参考になる．
# 
# - [Anacondaの公式サイト](https://www.anaconda.com/products/individual)にアクセスする
# - 下にスクロールし，"Anaconda Installers"から環境に応じたインストーラをダウンロードする
# - ダウンロードしたインストーラをクリックし，画面の指示に従う
#     - 途中で，`add Anaconda to the system Path environment variable`にチェックを入れてPathの設定を行う
# - Anaconda Navigatorが起動するか確認する

# ## 作業フォルダの作成
# データ分析では，様々なファイルを扱わなければならない．
# これらのファイルが自分のPC内のどこに保存されているかを把握しておかないと，ファイルを探すだけで時間を取られてしまい，時間の無駄である．
# データ分析を始める際にまず行うべきことは，PC内のフォルダやファイルを整理することである．
# 
# まず本講義専用の作業フォルダを作成する（名前は自分で分かれば何でも良い）．
# 作業フォルダの作成場所はできればOneDriveの中に作ることを推奨する（こうすれば，自動的にクラウド上にバックアップされる）．
# 
# ここでは，`ローカルディスク（C:）>ユーザー>username>OneDrive`の中に`情報処理の応用`という作業フォルダを作ったとする：
# ```
# [OneDrive]
#     - [デスクトップ]
#     - [ドキュメント]
#     ...
#     - [情報処理の応用]
# 
# ```

# 本講義で扱うファイルは全てこの`情報処理の応用`フォルダの中に保存する．
# `情報処理の応用`フォルダの中身は次のように章ごとのサブフォルダやレポート用のフォルダに分けておくと良い：
# ```
# [情報処理の応用]
#     - [chap_1]
#     - [chap_2]
#         - [sec_2-1]
#         - [sec_2-2]
#     ...
#     - [report]
#     - [others]
# ```

# ## Jupyter Labの運用
# 
# Anacondaをインストールすると，自動的にJupyter Labが使えるようになる．
# 本講義では演習の時間を設けるので，その際には以下のようにJupyter Labを運用する．

# ### Jupyter Labの起動
# 
# - Anaconda Navigatorを起動
#     - ［スタートメニュー］→［Anaconda Navigator (anaconda3)］
# - ［Jupyter Lab］をクリック

# ### .ipynbファイルの起動

# - `.ipynb`ファイルをダウンロードし，作業フォルダ内に保存する．
#     - 講義ノート上部のアイコンから`.ipynb`をクリック
#     - 自動保存された場合は[ダウンロード]フォルダ内に保存されているはず
# - Jupyter Lab左上のフォルダアイコンをクリックする．
# - .ipynbファイルを保存したフォルダに移動し，`.ipynb`ファイルをダブルクリックする．

# ## パス（Path）について
# 
# ### パスとは何か？
# 
# 自分のPCに保存されたファイルをJupyter Labに読み込んだり，逆にJupyter Labで作成した図などを自分のPCに保存するには，対象となるファイルの在り処，つまりアドレスが分からないといけない．
# このアドレスを指定する文字列のことをパス（Path）と呼ぶ．
# 
# Windowsの場合，パスはフォルダの階層構造を区切り文字`¥`（またはバックスラッシュ`\`）によって区切った形式で以下のように表される：
# 
# ```
# C:¥ユーザー¥ドキュメント
# ```

# フォルダの階層の区切りは`¥`（またはバックスラッシュ`\\`）によって表されており，`¥`の隣にはフォルダの名前が記載されている．
# 上の例は，Cドライブ（`C:`）の中にある`ユーザー`フォルダの中の`ドキュメント`フォルダのパスを表す．

# ### 相対パスと絶対パス
# パスには相対パスと絶対パスの2種類が存在する．

# **相対パス**
# 
# 現在自分がいるフォルダのことを**カレントディレクトリ**と呼ぶ．
# 通常は，Jupyter Labで開いている`.ipynb`ファイルが保存されているフォルダがカレントディレクトリとなる．
# 相対パスとはカレントディレクトリからの相対的な位置を示す方法で，例えば以下のように指定する：
# ```
# ./test.pdf
# ```
# ここで，先頭の`.`はカレントディレクトリ（`sec_1-3.ipynb`が保存されているフォルダ）を意味する文字である．
# よって，この場合はカレントディレクトリの中にある`test.pdf`というファイルを示すパスとなる．
# 
# 練習として，以下のコードを実行してみよう．
# カレントディレクトリの中に`test.pdf`というファイルができるはずである．

# In[7]:


fig, ax = plt.subplots(figsize=(3.5, 3))
x = np.arange(-np.pi, np.pi, 0.01)
ax.plot(x, np.sin(x))
ax.set_xlabel('横軸') # 横軸のラベル
ax.set_ylabel('縦軸') # 縦軸のラベル
ax.set_title('三角関数のグラフ', fontsize=12) # タイトル

fig.savefig('./test.pdf', bbox_inches="tight", pad_inches=0.2, transparent=True, dpi=300)


# **絶対パス**
# 
# 最も上の階層であるドライブ名（通常は`C:`）から始まるパスを**絶対パス**と呼ぶ．
# Windowsにおいて絶対パスを取得（コピー）するには以下のようにする：
# 
# - エクスプローラー上で対象のファイルやフォルダに対しshiftキーを押しながら右クリック
# - 「パスのコピー」を選択
# 
# 

# なお，Windows環境においてパスをコピーして貼り付けると
# ``
# C:\ユーザー\ユーザー名\OneDrive\情報処理の応用
# ``
# のように区切り文字がバックスラッシュ`\`または`¥`になるはずである．
# ところが，pythonではバックスラッシュ`\`と文字を組み合わせたエスケープシーケンスいう特別な文字列が存在し，例えば，`\n`は改行，`\t`はタブを表すエスケープシーケンスとなる．
# これにより，上の例の中にある`\t`の部分はパスの区切りではなくエスケープシーケンスとして認識され，エラーが出ることがある（特に，pythonでファイルの入出力を行うとき）．
# これを回避するには以下のように先頭に`r`を付ける
# ```
# r"C:\ユーザー\ユーザー名\OneDrive\情報処理の応用"
# ```
# これは，raw文字列と呼ばれ，""の中に指定した文字列をそのままの形で認識させることができる．

# 実際に，自分の好きなフォルダのパスをコピーし，`test2.pdf`という名前で図を保存してみよう．

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(3.5, 3))
x = np.arange(-np.pi, np.pi, 0.01)
ax.plot(x, np.sin(x))

fig.savefig(, bbox_inches="tight", pad_inches=0.2, transparent=True, dpi=300)


# ## Maplotlibの日本語対応
# 
# Matplotlibはグラフ作成のためのライブラリである．
# Matplotlibは標準で日本語に対応していないが，以下の2つの方法で日本語を出力することができる

# **1. matplotlibで使用する日本語フォントを設定する**

# pipでのインストールを行いたくない場合は，matplotlib用の日本語フォントその都度指定する方法もある．
# 以下のように，好きな日本語フォントを使用できるが，WindowsやMacによって使用可能なフォントが異なるので注意する．
# フォントにこだわりがなければ，Windowsの場合は'MS Gothic'，Macの場合は'Hiragino Sans'としておけば良い．

# In[8]:


# 日本語フォントの設定（Mac:'Hiragino Sans'，Windows: 'MS Gothic'）
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'MS Gothic'


# **2. japanize_matplotlib を利用する（pipでインストールする必要がある）**
# 
# japanize_matplotlibはPythonのモジュールなので，最初にインストールしておけば，あとは他のモジュールと同じように`import japanize_matplotlib`とするだけで日本語が使用可能になる．
# ただし，使用可能なフォントはIPAexゴシックだけなので，フォントにこだわりたい場合は2.の方法をおすすめする．
# 
# **japanize_matplotlibのインストール（詳しくは[こちら](https://pypi.org/project/japanize-matplotlib/)）**
# 
# - ターミナルを開いて以下のコマンドを実行し，AnacondaのインストールされているフォルダのPathを取得する
#     ```
#     conda info -e
#     ```
# - `*`の右に表示された文字列（フォルダのパス）をコピーして以下を実行
#   ```zsh
#   activate "フォルダのパス"
#   ```
# - 以下のコマンドを実行してインストールする
#     ```zsh
#     pip install japanize-matplotlib
#     ```
