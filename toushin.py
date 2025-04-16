# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 01:05:03 2025

@author: my199
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

def get_data(ISIN, FUND):
    BASEURL = "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?"
    ISINcd  = "isinCd=" + ISIN
    FUNDcd  = "associFundCd=" + FUND
    DOWNURL = BASEURL + ISINcd + "&" + FUNDcd
    DATE_PARSE = lambda date: datetime.strptime(date, "%Y年%m月%d日")
    df = pd.read_csv(DOWNURL, engine="python", encoding="shift-jis", index_col="年月日", parse_dates=True, date_parser=DATE_PARSE)
    return df

def join_data(df_part, df_join, KEYWORD, str_date, end_date):
    df_part_fil = df_part.loc[(df_part.index >= str_date) & (df_part.index <= end_date), :]
    df_part_fil = df_part_fil.rename(columns={"基準価額(円)": KEYWORD})[[KEYWORD]]
    df_join = pd.merge(df_join, df_part_fil, left_index=True, right_index=True, how="outer") if df_join is not None else df_part_fil
    return df_join

st.title("投資信託のリスク・リターン分析とシミュレーション")

num_funds = st.number_input("保有投資信託の数を入力してください", min_value=1, max_value=10, value=1)

inputs = []
for i in range(num_funds):
    with st.expander(f"ファンド{i+1}の情報を入力"):
        isin = st.text_input(f"ISINコード{i+1}", key=f"isin{i}")
        fundcode = st.text_input(f"ファンドコード{i+1}", key=f"fund{i}")
        inputs.append((isin, fundcode))

if st.button("データを取得して分析を開始"):
    str_date = "2015-04-01"
    end_date = "2025-03-31"
    df_join = None
    asset_names = []

    for idx, (isin, fundcode) in enumerate(inputs):
        if isin and fundcode:
            try:
                df = get_data(isin, fundcode)
                name = f"ファンド{idx+1}"
                df_join = join_data(df, df_join, name, str_date, end_date)
                asset_names.append(name)
            except Exception as e:
                st.error(f"{name} のデータ取得に失敗しました: {e}")
        else:
            st.warning(f"ファンド{idx+1}の情報が未入力です。")

    if df_join is not None:
        st.subheader("基準価額（過去）")
        st.line_chart(df_join)

        st.subheader("リターン（年率）")
        returns = []
        for col in df_join.columns:
            data = df_join[col].dropna()
            length = len(data) - 1
            if length <= 0:
                returns.append(np.nan)
                continue
            fund_return = 100 * ((data.pct_change()[1:] + 1).prod() ** (250 / length) - 1)
            returns.append(fund_return)
        df_returns = pd.DataFrame(returns, index=df_join.columns, columns=["リターン（年率）"])
        st.dataframe(df_returns.style.format("{:.2f}%"))

        st.subheader("リスク（年率）")
        df_vola = (df_join.pct_change()[1:] * 100).std() * (250 ** 0.5)
        df_vola = pd.DataFrame(df_vola, columns=["リスク（年率）"])
        st.dataframe(df_vola.style.format("{:.2f}%"))

        if len(df_join.columns) > 1:
            st.subheader("相関行列")
            st.dataframe(df_join.corr().style.background_gradient(cmap="coolwarm").format("{:.2f}"))

        # --- ここで将来シミュレーション ---
        st.subheader("将来シミュレーション（モンテカルロ）")
        n_years = st.slider("シミュレーション期間（年）", 1, 30, 10)
        n_scenarios = st.slider("シナリオ数", 100, 1000, 300)

        sim_results = {}
        for col in df_join.columns:
            mu = df_returns.loc[col].values[0] / 100
            sigma = df_vola.loc[col].values[0] / 100
            start_price = df_join[col].dropna()[-1]
            sim_data = np.zeros((n_scenarios, n_years + 1))
            sim_data[:, 0] = start_price
            for t in range(1, n_years + 1):
                sim_data[:, t] = sim_data[:, t-1] * np.exp(np.random.normal(mu - 0.5 * sigma ** 2, sigma, n_scenarios))
            sim_results[col] = sim_data

        for col, sim_data in sim_results.items():
            df_sim = pd.DataFrame(sim_data.T)
            st.line_chart(df_sim)
