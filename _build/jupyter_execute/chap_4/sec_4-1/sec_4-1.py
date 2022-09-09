#!/usr/bin/env python
# coding: utf-8

# In[74]:


# カレントディレクトリの変更（自分の作業フォルダのパスを設定）
os.chdir('/Users/narizuka/GoogleDrive/My Drive/document/講義/立正/情報処理の応用/')
# os.chdir("G:\\マイドライブ\\document\\講義\\立正\\情報処理の応用")


# 使用するモジュールのimport

# In[75]:


import sys, os
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
from pandas import DataFrame
from scipy import optimize
import material.fit_func as ff


# In[76]:


pd.set_option('max_rows', 20)  # 表示する行数
pd.set_option('precision', 4)  # 小数点以下の表示桁
np.set_printoptions(suppress=True, precision=4)
get_ipython().run_line_magic('precision', '4  # 小数点以下の表示桁')


# google colab を使う際のセッティング

# In[ ]:


# matplotlibで日本語表示
get_ipython().system('pip install japanize-matplotlib')
import matplotlib.pyplot as plt
import japanize_matplotlib


# In[ ]:


# google driveをマウントする
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# カレントディレクトリを変更（自分の作業フォルダのパスを設定）
os.chdir('/content/drive/My Drive/document/講義/立正/情報処理の応用/')


# jupyter lab のcssスタイルを変更（必要な場合だけ）

# In[77]:


from IPython.core.display import display, HTML
display(HTML("<style>.jp-Cell { width:100% !important; margin: 0 auto; }</style>"))
with open('./material/variables.css') as f: 
    css = f.read().replace(';', ' !important;')
display(HTML('<style type="text/css">%s</style>'%css))


# # 二項分布から正規分布へ

# ## 二項分布

# In[13]:


fig, ax = plt.subplots()
k = np.arange(0, 20, 1)
ax.plot(k, ff.bi(k, 100, 0.01), '-o', mfc='w', ms=5, label='$n=100, p=0.01$')
ax.plot(k, ff.bi(k, 100, 0.04), '-o', mfc='w', ms=5, label='$n=100, p=0.04$')
ax.plot(k, ff.bi(k, 100, 0.08), '-o', mfc='w', ms=5, label='$n=100, p=0.08$')
ax.set_xlabel('$x$', fontsize=15)
ax.set_ylabel('$f(x)$', fontsize=15)
ax.legend(numpoints=1, fontsize=10, loc='upper right', frameon=True)

# fig.savefig(fpath+'binom_p.pdf', bbox_inches="tight", pad_inches=0.2, transparent=True)


# In[14]:


fig, ax = plt.subplots()
k = np.arange(0, 20, 1)
ax.plot(k, ff.bi(k, 5, 0.2), '-o', mfc='w', ms=5, label='$n=5, p=0.2$')
ax.plot(k, ff.bi(k, 10, 0.2), '-o', mfc='w', ms=5, label='$n=10, p=0.2$')
ax.plot(k, ff.bi(k, 30, 0.2), '-o', mfc='w', ms=5, label='$n=30, p=0.2$')
ax.plot(k, ff.bi(k, 50, 0.2), '-o', mfc='w', ms=5, label='$n=50, p=0.2$')
ax.set_xlim(0, 20); ax.set_ylim(0, 0.5)
ax.set_xlabel('$x$', fontsize=15)
ax.set_ylabel('$f(x)$', fontsize=15)
ax.legend(numpoints=1, fontsize=10, loc='upper right', frameon=True)

# fig.savefig(fpath+'binom_n.pdf', bbox_inches="tight", pad_inches=0.2, transparent=True)


# In[15]:


fig, ax = plt.subplots()
k = np.arange(0, 20, 1)
ax.plot(k/5, 5*ff.bi(k, 5, 0.2), '-o', mfc='w', ms=5, label='$n=5, p=0.2$')
ax.plot(k/10, 10*ff.bi(k, 10, 0.2), '-o', mfc='w', ms=5, label='$n=10, p=0.2$')
ax.plot(k/30, 30*ff.bi(k, 30, 0.2), '-o', mfc='w', ms=5, label='$n=30, p=0.2$')
ax.plot(k/50, 50*ff.bi(k, 50, 0.2), '-o', mfc='w', ms=5, label='$n=50, p=0.2$')
ax.set_xlim(0, 1); ax.set_ylim(0, 8)
ax.set_xlabel('$t=x/n$', fontsize=15)
ax.set_ylabel('$g(t)$', fontsize=15)
ax.legend(numpoints=1, fontsize=10, loc='upper right', frameon=True)

# fig.savefig(fpath+'binom_n2.pdf', bbox_inches="tight", pad_inches=0.2, transparent=True)


# ## 正規分布

# In[16]:


fig, ax = plt.subplots()
x = np.arange(-5, 5, 0.01)
ax.plot(x, ff.nm(x, 0, 0.5), '-', mfc='w', ms=5, label='$\mu=0, \sigma=0.5$')
ax.plot(x, ff.nm(x, 0, 1.0), '-', mfc='w', ms=5, label='$\mu=0, \sigma=1.0$')
ax.plot(x, ff.nm(x, 0, 2.0), '-', mfc='w', ms=5, label='$\mu=0, \sigma=2.0$')
# ax.set_xlim(0, 1); ax.set_ylim(0, 8)
ax.set_xlabel('$x$', fontsize=15)
ax.set_ylabel('$f(x)$', fontsize=15)
ax.legend(numpoints=1, fontsize=8, loc='upper right', frameon=True)

# fig.savefig(fpath+'normal.pdf', bbox_inches="tight", pad_inches=0.2, transparent=True)


# In[17]:


fig, ax = plt.subplots()
x = np.arange(-5, 8, 0.01)
ax.plot(x, ff.nm(x, 0, 0.5), '-', mfc='w', ms=5, label='$\mu=0, \sigma=0.5$')
ax.plot(x, ff.nm(x, 1, 1.0), '-', mfc='w', ms=5, label='$\mu=1, \sigma=1.0$')
ax.plot(x, ff.nm(x, 2, 2.0), '-', mfc='w', ms=5, label='$\mu=2, \sigma=2.0$')
# ax.set_xlim(0, 1); ax.set_ylim(0, 8)
ax.set_xlabel('$x$', fontsize=15)
ax.set_ylabel('$f(x)$', fontsize=15)
ax.legend(numpoints=1, fontsize=8, loc='upper right', frameon=True)

# fig.savefig(fpath+'normal2.pdf', bbox_inches="tight", pad_inches=0.2, transparent=True)


# ## 標準化と標準正規分布

# 正規分布の相補累積分布関数は
# $$
#     P(X \geq x) = \int_{x}^{\infty} \frac{1}{\sqrt{2\pi} \sigma} \exp \left[ - \frac{(x-\mu)^{2}}{2\sigma^{2}} \right] dx
# $$
# と表される．
# これは，誤差関数
# $$
#     \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} \mathrm{e}^{-t^2} dt
# $$
# を用いると，
# $$
#     P(X \geq x) = \frac{1}{2} \left[1 - \mathrm{erf}\left(\frac{x-\mu}{\sqrt{2}\sigma} \right) \right]
# $$
# と表される．

# In[98]:


fig, ax = plt.subplots(figsize=(4, 3))
x = np.arange(-5, 5, 0.01)
ax.plot(x, ff.nm(x, 0, 1), '-', mfc='w', ms=5)
x2 = np.arange(1, 5, 0.01)
plt.fill_between(x2, ff.nm(x2, 0, 1), facecolor='gray', alpha=0.5)
ax.set_xlim(-3.5, 3.5); ax.set_ylim(0, 0.45)
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$f(z)$', fontsize=15)
# ax.legend(numpoints=1, fontsize=8, loc='upper right', frameon=True)

# fig.savefig('figure/normal3.pdf', bbox_inches="tight", pad_inches=0.2, transparent=True)


# ### 標準正規分布の上側確率の計算
# $$
#     P(-a \leq Z \leq a)  = 1 - 2P(Z \geq a) = \mathrm{erf}(x/\sqrt{2})
# $$

# In[18]:


from scipy.special import erf


# In[22]:


# a=1
erf(1/sqrt(2))


# In[23]:


# a=2
erf(2/sqrt(2))


# In[24]:


# a=3
erf(3/sqrt(2))


# # 実例：視聴率調査の仕組みは？

# ## STEP1: Problem
# - 2016年10月31日時点で関東地区の世帯数は約1800万世帯である．
# - これらの世帯の中で，ある番組を見ている世帯の割合を表したものが番組視聴率である．
# - 通常，全世帯の視聴率を完全に把握するには全世帯を調査する必要があるが，それは現実的ではない．
# - そこで，一部の世帯だけを抽出し（これを標本と呼ぶ），そこでの視聴率調査から全世帯の視聴率を推定する方法が取られる．
# - 実際，ビデオリサーチ社の視聴率調査において調査対象となる世帯数は関東地区で900世帯となっている．
# - では，どのような方法で900世帯のデータから全体の視聴率を推定しているのだろうか？

# ## STEP 2: Plan
# - 以下のような視聴率調査を模した模擬実験を考え，世帯数と視聴率調査の正確性の関係を調べる．
# - まず，黒玉を「番組を見た世帯」，白玉を「番組を見ていない世帯」とする．
# - 黒玉は20個，白玉は60個用意して箱の中に入れる．
# - すなわち，視聴率の理論値は20/80=25\%となる．
# - 実験では，まず，箱の中から$ 4 $個の玉を取り出すことを200回繰り返し，黒玉の比率（視聴率）のヒストグラムを作成する．
# - 次に，標本サイズが$ n=12,15,30 $の場合に対して同様のことを繰り返し，ヒストグラムの変化を見る．

# ## STEP 3: Data
# - $ n=4,12,15,30 $の場合に模擬実験を200回繰り返した結果，以下のような結果を得た．

# In[9]:


D4 = DataFrame({'rate': [0, 25, 50, 75],
                'freq': [70, 80, 40, 10]})
D12 = DataFrame({'rate': [0, 8, 17, 25, 33, 42, 50],
                 'freq': [5, 30, 50, 45, 50, 15, 5]})
D15 = DataFrame({'rate': [0, 7, 13, 20, 27, 33, 40, 47, 53],
                 'freq': [3, 6, 30, 42, 54, 36, 18, 6, 5]})
D30 = DataFrame({'rate': [3, 7, 10, 13, 17, 20, 23, 27, 30, 33, 37, 40, 43],
                 'freq': [2, 1, 3, 5, 7, 28, 46, 56, 18, 14, 10, 8, 2]})


# ## STEP 4: Analysis
# #### 実習
# - 実験で得られたデータを用いて，各 $ n $ に対する黒玉比率のヒストグラムを作成せよ．
# - $ n $ の増加に応じてヒストグラムの形がどのように変化するか確認せよ．
# - 各 $ n $ に対して黒玉比率の標本平均と標準偏差を計算せよ．

# In[24]:


fig, ax = plt.subplots()
ax.bar(D4['rate'], D4['freq'], width=1, color='b')
ax.set_xlabel('黒玉比率 [%]')
ax.set_ylabel('頻度')


# In[25]:


# n=12
fig, ax = plt.subplots()
ax.bar(D12['rate'], D12['freq'], width=1, color='b')
ax.set_xlabel('黒玉比率 [%]')
ax.set_ylabel('頻度')


# In[26]:


# n=15
fig, ax = plt.subplots()
ax.bar(D15['rate'], D15['freq'], width=1, color='b')
ax.set_xlabel('黒玉比率 [%]')
ax.set_ylabel('頻度')


# In[27]:


# n=30
fig, ax = plt.subplots()
ax.bar(D30['rate'], D30['freq'], width=1, color='b')
ax.set_xlabel('黒玉比率 [%]')
ax.set_ylabel('頻度')


# #### 各 $ n $ に対する黒玉比率の標本平均と標準偏差

# In[ ]:





# ## STEP 5: Conclusion
# #### 実習
# - 実験の結果得られたヒストグラムについて，大数の法則の観点から考察せよ．
# - 実験の結果得られたヒストグラムについて，中心極限定理の観点から考察せよ．
# - 実験全体を踏まえ，視聴率調査の仕組みについて考察せよ．
