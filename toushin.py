# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 01:05:03 2025

@author: my199
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# ファンドデータ取得
def get_data(ISIN, FUND):
    BASEURL = "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?"
    ISINcd  = "isinCd=" + ISIN
    FUNDcd  = "associFundCd=" + FUND
    DOWNURL = BASEURL + ISINcd + "&" + FUNDcd
    DATE_PARSE = lambda date: datetime.strptime(date, "%Y年%m月%d日")
    df = pd.read_csv(DOWNURL, engine="python", encoding="shift-jis", index_col="年月日", parse_dates=True, date_parser=DATE_PARSE)

    # ファンド名取得
    name_row = pd.read_csv(DOWNURL, engine="python", encoding="shift-jis", nrows=1)
    name = name_row.columns[1] if len(name_row.columns) > 1 else "ファンド名不明"

    return df, name

# データ結合
def join_data(df_part, df_join, KEYWORD, str_date, end_date):
    df_part_fil = df_part.loc[(df_part.index >= str_date) & (df_part.index <= end_date), :]
    df_part_fil = df_part_fil.rename(columns={"基準価額(円)": KEYWORD})[[KEYWORD]]
    df_join = pd.merge(df_join, df_part_fil, left_index=True, right_index=True, how="outer") if df_join is not None else df_part_fil
    return df_join

st.title("投資信託のシミュレーション分析")

num_funds = st.number_input("保有している投資信託の本数を入力", min_value=1, max_value=5, value=2)

isin_list = []
fundcode_list = []
for i in range(num_funds):
    isin = st.text_input(f"ファンド{i+1}のISINコード", value="JP90C000H1T1" if i == 0 else "JP90C000ATX6")
    fundcode = st.text_input(f"ファンド{i+1}のファンドコード", value="0331418A" if i == 0 else "39312149")
    isin_list.append(isin)
    fundcode_list.append(fundcode)

# データ取得・結合
dict_assets = {}
df_join = None
for i in range(num_funds):
    df, name = get_data(isin_list[i], fundcode_list[i])
    dict_assets[name] = df

# 結合期間の最大・最小
min_date = max(df.index.min() for df in dict_assets.values()).date()
max_date = min(df.index.max() for df in dict_assets.values()).date()

str_date = st.slider("分析開始日を選択", min_value=min_date, max_value=max_date, value=min_date)
str_date = pd.to_datetime(str_date)
end_date = pd.to_datetime(max_date)

for name, df in dict_assets.items():
    df_join = join_data(df, df_join, name, str_date, end_date)

# 欠損処理
df_join = df_join.fillna(method="ffill").dropna()

# リターン計算（年率）
returns = []
for i in range(df_join.shape[1]):
    length = df_join.iloc[:, i].count() - 1
    days = 250
    r = 100 * ((df_join.iloc[:, i].pct_change()[1:] + 1).prod() ** (days / length) - 1)
    returns.append(r)
returns = np.array(returns)
df_returns = pd.DataFrame(returns, index=df_join.columns, columns=["リターン（年率）"])

# リスク計算（年率）
df_vola = (df_join.pct_change()[1:] * 100).std() * (days ** 0.5)
df_vola = pd.DataFrame(df_vola, columns=["リスク（年率）"])

# 相関
df_corr = df_join.pct_change().corr()

# モンテカルロシミュレーション（シンプル）
num_sim = 300
years = 20
sim_results = []
initial = 100

col1, col2 = st.columns(2)
col1.dataframe(df_returns)
col2.dataframe(df_vola)

if df_join.shape[1] == 1:
    fund_name = df_join.columns[0]
    mu = df_returns.loc[fund_name].values[0] / 100
    sigma = df_vola.loc[fund_name].values[0] / 100

    for _ in range(num_sim):
        path = [initial]
        for _ in range(years):
            ret = np.random.normal(mu, sigma)
            path.append(path[-1] * (1 + ret))
        sim_results.append(path)

    sim_array = np.array(sim_results)
    percentiles = np.percentile(sim_array, [25, 50, 75], axis=0)

    # 描画
    fig, ax = plt.subplots()
    for i in range(num_sim):
        ax.plot(sim_array[i], color='lightgray', linewidth=0.5, alpha=0.5)

    ax.plot(percentiles[0], 'b--', label="25%")
    ax.plot(percentiles[1], 'r-', label="50%")
    ax.plot(percentiles[2], 'b--', label="75%")
    ax.set_title(f"「{fund_name}」ファンドのモンテカルロシミュレーション")
    ax.set_xlabel("年数")
    ax.set_ylabel("資産価値")
    ax.legend()

    st.pyplot(fig)

    # 50%ラインから年率リターン算出
    end_val = percentiles[1][-1]
    ann_return = ((end_val / initial) ** (1 / years)) - 1
    st.metric(label="50%ラインの年率リターン", value=f"{ann_return*100:.2f} %")
else:
    st.write("相関係数：")
    st.dataframe(df_corr)
