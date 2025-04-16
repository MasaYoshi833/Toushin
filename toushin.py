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

def get_data(ISIN, FUND):
    BASEURL = "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?"
    ISINcd  = "isinCd=" + ISIN
    FUNDcd  = "associFundCd=" + FUND
    DOWNURL = BASEURL + ISINcd + "&" + FUNDcd
    DATE_PARSE = lambda date: datetime.strptime(date, "%Y年%m月%d日")
    df = pd.read_csv(DOWNURL, engine="python", encoding="shift-jis", index_col="年月日", parse_dates=True, date_parser=DATE_PARSE)
    df_name = pd.read_csv(DOWNURL, engine="python", encoding="shift-jis", nrows=1)
    name = df_name.columns[0]
    return df, name

def join_data(df_part, df_join, KEYWORD):
    df_part = df_part.rename(columns={"基準価額(円)": KEYWORD})[[KEYWORD]]
    df_join = pd.merge(df_join, df_part, left_index=True, right_index=True, how="outer") if df_join is not None else df_part
    return df_join

st.title("投資信託分析＆将来シミュレーションアプリ")

num_funds = st.number_input("保有している投資信託の数", min_value=1, max_value=10, value=1)
inputs = []

for i in range(num_funds):
    with st.expander(f"ファンド{i+1}の情報"):
        isin = st.text_input(f"ISINコード{i+1}", key=f"isin{i}")
        fundcode = st.text_input(f"ファンドコード{i+1}", key=f"fund{i}")
        inputs.append((isin, fundcode))

if st.button("データ取得・分析開始"):
    df_join = None
    fund_names = []
    raw_dfs = {}
    for idx, (isin, fundcode) in enumerate(inputs):
        if isin and fundcode:
            try:
                df_raw, name = get_data(isin, fundcode)
                df_join = join_data(df_raw, df_join, name)
                raw_dfs[name] = df_raw
                fund_names.append(name)
                st.success(f"✅ あなたの保有している「{name}」ファンドのデータを取得しました。")
            except Exception as e:
                st.error(f"❌ データ取得失敗: {e}")
        else:
            st.warning(f"⚠️ ファンド{idx+1}の情報が未入力です。")

    if df_join is not None:
        df_join = df_join.sort_index()
        min_date = df_join.index.min()
        max_date = df_join.index.max()
        str_date = st.slider("分析開始日を選択", min_value=min_date, max_value=max_date, value=min_date)
        df_filtered = df_join[df_join.index >= str_date]

        st.subheader("基準価額の推移")
        st.line_chart(df_filtered)

        st.subheader("年率リターン")
        returns = []
        for col in df_filtered.columns:
            data = df_filtered[col].dropna()
            length = len(data) - 1
            r = 100 * ((data.pct_change()[1:] + 1).prod() ** (250 / length) - 1)
            returns.append(r)
        df_ret = pd.DataFrame(returns, index=df_filtered.columns, columns=["リターン（年率）"])
        st.dataframe(df_ret.style.format("{:.2f}%"))

        st.subheader("年率リスク")
        df_vola = (df_filtered.pct_change()[1:] * 100).std() * (250 ** 0.5)
        df_vola = pd.DataFrame(df_vola, columns=["リスク（年率）"])
        st.dataframe(df_vola.style.format("{:.2f}%"))

        if len(df_filtered.columns) > 1:
            st.subheader("相関行列")
            st.dataframe(df_filtered.corr().style.background_gradient(cmap="coolwarm").format("{:.2f}"))

        # --- 将来シミュレーション ---
        st.subheader("将来シミュレーション（モンテカルロ法）")
        n_years = st.slider("シミュレーション期間（年）", 1, 30, 10)
        n_scenarios = st.slider("シナリオ数", 100, 1000, 300)

        for col in df_filtered.columns:
            st.markdown(f"### 📈 {col}")
            mu = df_ret.loc[col].values[0] / 100
            sigma = df_vola.loc[col].values[0] / 100
            S0 = df_filtered[col].dropna()[-1]

            sim_data = np.zeros((n_scenarios, n_years + 1))
            sim_data[:, 0] = S0
            for t in range(1, n_years + 1):
                sim_data[:, t] = sim_data[:, t-1] * np.exp(np.random.normal(mu - 0.5 * sigma ** 2, sigma, n_scenarios))

            # 中央/四分位シナリオ
            median = np.percentile(sim_data, 50, axis=0)
            p25 = np.percentile(sim_data, 25, axis=0)
            p75 = np.percentile(sim_data, 75, axis=0)

            # グラフ描画
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(min(50, n_scenarios)):
                ax.plot(sim_data[i], color='lightgrey', linewidth=0.8, alpha=0.5)
            ax.plot(median, color='red', label='50%（中央値）', linewidth=2)
            ax.plot(p25, color='blue', linestyle='--', label='25%', linewidth=1.5)
            ax.plot(p75, color='blue', linestyle='--', label='75%', linewidth=1.5)
            ax.set_title(f"{col} 将来価格シナリオ")
            ax.set_xlabel("年")
            ax.set_ylabel("価格")
            ax.legend()
            st.pyplot(fig)

            # 年率リターン・リスク（中央値から計算）
            returns_median = pd.Series(median).pct_change().dropna()
            median_return = ((returns_median + 1).prod()) ** (1 / len(returns_median)) - 1
            median_risk = returns_median.std() * (250 ** 0.5)
            st.markdown(f"📊 中央シナリオの年率リターン: **{median_return * 100:.2f}%**")
            st.markdown(f"📉 中央シナリオの年率リスク: **{median_risk * 100:.2f}%**")
