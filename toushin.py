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

# データ取得関数
def get_data(ISIN, FUND):
    BASEURL = "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?"
    ISINcd  = "isinCd=" + ISIN
    FUNDcd  = "associFundCd=" + FUND
    DOWNURL = BASEURL + ISINcd + "&" + FUNDcd
    DATE_PARSE = lambda date: datetime.strptime(date, "%Y年%m月%d日")
    
    # データ本体
    df = pd.read_csv(DOWNURL, engine="python", encoding="shift-jis", index_col="年月日", parse_dates=True, date_parser=DATE_PARSE)
    
    # ファンド名取得（1行目の2列目がファンド名）
    raw = pd.read_csv(DOWNURL, engine="python", encoding="shift-jis", nrows=1)
    fund_name = raw.columns[1] if len(raw.columns) > 1 else "ファンド名不明"
    
    return df, fund_name

# タイトル
st.title("投資信託シミュレーション")

# 入力
isin = st.text_input("ISINコード", value="JP90C000H1T1")
fundcode = st.text_input("ファンドコード", value="0331418A")

if isin and fundcode:
    try:
        df, fund_name = get_data(isin, fundcode)
        st.subheader(f"あなたの保有しているファンドは・・・")
        
        # 期間選択
        min_date = df.index.min().date()
        max_date = df.index.max().date()
        str_date = st.slider("分析開始日を選択", min_value=min_date, max_value=max_date, value=min_date)
        str_date = pd.to_datetime(str_date)
        df_filtered = df[df.index >= str_date]
        df_filtered = df_filtered.fillna(method="ffill").dropna()
        
        # 年率リターン・リスク計算
        days = 250
        returns = df_filtered["基準価額(円)"].pct_change().dropna()
        length = len(returns)
        ann_return = 100 * ((returns + 1).prod() ** (days / length) - 1)
        ann_risk = 100 * returns.std() * (days ** 0.5)
        
        st.metric("年率リターン", f"{ann_return:.2f} %")
        st.metric("年率リスク", f"{ann_risk:.2f} %")

        # モンテカルロシミュレーション
        num_sim = 300
        years = 20
        mu = ann_return / 100
        sigma = ann_risk / 100
        initial = 100
        sim_results = []

        for _ in range(num_sim):
            path = [initial]
            for _ in range(years):
                ret = np.random.normal(mu, sigma)
                path.append(path[-1] * (1 + ret))
            sim_results.append(path)

        sim_array = np.array(sim_results)
        percentiles = np.percentile(sim_array, [25, 50, 75], axis=0)

        # グラフ描画
        fig, ax = plt.subplots()
        for i in range(num_sim):
            ax.plot(sim_array[i], color='lightgray', linewidth=0.5, alpha=0.5)

        ax.plot(percentiles[0], 'b--', label="25 Percentile")
        ax.plot(percentiles[1], 'r-', label="50 Percentile(Median)")
        ax.plot(percentiles[2], 'b--', label="75 Percentile")
        ax.set_title(f"（{years}years Monte Carlo Simulation）")
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

        # 50%ラインから年率換算リターン
        end_val = percentiles[1][-1]
        ann_sim_return = ((end_val / initial) ** (1 / years)) - 1
        st.success(f"中央値シナリオの年率リターン：{ann_sim_return * 100:.2f} %")
        
    except Exception as e:
        st.error(f"データの取得に失敗しました。入力内容またはネットワークをご確認ください。\n\n詳細：{e}")

