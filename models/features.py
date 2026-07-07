"""构造多因子特征矩阵:技术面 + 估值面 + 基本面 + 大盘联动

供 rf_model.py 和 lstm_model.py 共用。返回的 DataFrame 按 date 升序排列,
每一行包含"当天已知信息"计算出的因子,以及"下一交易日"的标签列(用于监督学习)。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
from data.fetch_data import load_history
from data.fetch_fundamentals import load_fundamentals, align_to_daily
from data.fetch_index import load_index

TECHNICAL_FEATURES = [
    "momentum_5", "momentum_10", "momentum_20",
    "volatility_10", "volatility_20",
    "volume_change_5",
    "ma5_dev", "ma10_dev", "ma20_dev", "ma60_dev",
    "turnover_rate",
]

VALUATION_FEATURES = ["pe_ttm", "pb_mrq", "ps_ttm", "pcf_ncf_ttm", "pe_pct_rank_252"]

FUNDAMENTAL_FEATURES = ["roeAvg", "npMargin", "gpMargin", "epsTTM", "YOYNI", "YOYEPSBasic"]

MARKET_FEATURES = ["excess_return_5", "excess_return_20", "beta_60"]

ALL_FEATURES = TECHNICAL_FEATURES + VALUATION_FEATURES + FUNDAMENTAL_FEATURES + MARKET_FEATURES


def _pct_rank(window):
    if len(window) <= 1:
        return 0.5
    return float((window < window[-1]).sum()) / (len(window) - 1)


def build_feature_frame():
    hist = load_history()
    fundamentals = load_fundamentals()
    index_df = load_index()

    df = hist.merge(index_df[["date", "index_close", "index_pct_change"]], on="date", how="left")

    # ---------- 技术面因子 ----------
    df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    daily_return = df["close"].pct_change()
    df["volatility_10"] = daily_return.rolling(10).std()
    df["volatility_20"] = daily_return.rolling(20).std()

    df["volume_change_5"] = df["volume"] / df["volume"].rolling(5).mean() - 1

    for w in (5, 10, 20, 60):
        ma = df["close"].rolling(w).mean()
        df[f"ma{w}_dev"] = df["close"] / ma - 1

    # ---------- 估值面因子 ----------
    df["pe_pct_rank_252"] = df["pe_ttm"].rolling(252, min_periods=60).apply(_pct_rank, raw=True)

    # ---------- 基本面因子(按公告日前向填充对齐,避免未来数据泄露) ----------
    aligned_fund = align_to_daily(df["date"], fundamentals)
    df = df.merge(aligned_fund, on="date", how="left")
    for col in FUNDAMENTAL_FEATURES:
        df[col] = df[col].ffill()

    # ---------- 大盘联动因子 ----------
    df["excess_return_5"] = (
        df["close"].pct_change(5) - df["index_close"].pct_change(5)
    )
    df["excess_return_20"] = (
        df["close"].pct_change(20) - df["index_close"].pct_change(20)
    )
    cov60 = df["pct_change"].rolling(60).cov(df["index_pct_change"])
    var60 = df["index_pct_change"].rolling(60).var()
    df["beta_60"] = cov60 / var60

    # ---------- 标签(下一交易日) ----------
    df["next_close"] = df["close"].shift(-1)
    df["next_direction"] = np.where(df["next_close"] >= df["close"], 1, -1)
    df["next_pct_change"] = (df["next_close"] - df["close"]) / df["close"] * 100

    return df


if __name__ == "__main__":
    config.fix_console_encoding()
    df = build_feature_frame()
    print(f"特征矩阵形状: {df.shape}")
    print(f"因子列: {ALL_FEATURES}")
    print("\n缺失值统计(仅显示因子列):")
    print(df[ALL_FEATURES].isna().sum())
    print("\n最近5行因子样例:")
    print(df[["date"] + ALL_FEATURES].tail(5).to_string(index=False))
