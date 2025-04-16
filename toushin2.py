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

    df = pd.read_csv(DOWNURL, engine="python", encoding="shift-jis", index_col="年月日", parse_dates=True, date_parser=DATE_PARSE)
    raw = pd.read_csv(DOWNURL, engine="python", encoding="shift-jis", nrows=1)
    fund_name = raw.columns[1] if len(raw.columns) > 1 else "ファンド名不明"
    return df, fund_name

st.title("投資信託シミュレーション")

# 入力
isin = st.text_input("ISINコード", value="JP90C000H1T1")
fundcode = st.text_input("ファンドコード", value="0331418A")

# 実行ボタン
run_sim = st.button("シミュレーションを実行")

if run_sim and isin and fundcode:
    try:
        df, fund_name = get_data(isin, fundcode)
        st.subheader("あなたの保有しているファンドは・・・")

        # 日付調整
        df = df.fillna(method="ffill").dropna()
        df = df[df.index <= "2025-03-31"]

        # 月単位の選択肢生成
        df_months = df.resample("MS").first()
        month_options = df_months.index.to_list()
        month_strs = [d.strftime("%Y-%m") for d in month_options]
        selected_month_str = st.selectbox("分析開始月を選択", month_strs)
        str_date = pd.to_datetime(selected_month_str + "-01")

        # フィルタリング
        df_filtered = df[df.index >= str_date]
        returns = df_filtered["基準価額(円)"].pct_change().dropna()

        # 年率リターン・リスク
        days = 250
        length = len(returns)
        ann_return = 100 * ((returns + 1).prod() ** (days / length) - 1)
        ann_risk = 100 * returns.std() * (days ** 0.5)

        st.metric("年率リターン", f"{ann_return:.2f} %")
        st.metric("年率リスク", f"{ann_risk:.2f} %")

        # モンテカルロ
        num_sim = 1000
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

        # 指定パーセンタイルに近いシナリオのインデックスを探す（最終値）
        final_values = sim_array[:, -1]
        idx_25 = np.argmin(np.abs(final_values - percentiles[0, -1]))
        idx_50 = np.argmin(np.abs(final_values - percentiles[1, -1]))
        idx_75 = np.argmin(np.abs(final_values - percentiles[2, -1]))

        # グラフ描画（対象パスのみ表示）
        fig, ax = plt.subplots()
        ax.plot(sim_array[idx_25], 'b--', label="25%")
        ax.plot(sim_array[idx_50], 'r-', label="50%(Median)")
        ax.plot(sim_array[idx_75], 'b--', label="75%")

        # Y軸の上限調整（中央値の最終値の85%）
        ymax = sim_array[idx_50, -1] * 1.85
        ax.set_ylim([0, ymax])

        ax.set_title("Monte Carlo Simulation（20years）")
        ax.set_xlabel("Years")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

        # 中央シナリオの年率リターンとリスク
        sim_median = sim_array[idx_50]
        ann_sim_return = ((sim_median[-1] / sim_median[0]) ** (1 / years)) - 1
        yearly_returns = np.diff(sim_median) / sim_median[:-1]
        ann_sim_risk = np.std(yearly_returns) * (1 ** 0.5)  # 1年ごとの変動なので√1

        st.success(
            f"中央値シナリオの年率リターン：{ann_sim_return * 100:.2f} %\n"
            f"中央値シナリオの年率リスク：{ann_sim_risk * 100:.2f} %"
        )

    except Exception as e:
        st.error(f"データの取得に失敗しました。入力内容またはネットワークをご確認ください。\n\n詳細：{e}")

