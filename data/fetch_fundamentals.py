"""拉取季度财务数据(盈利能力+成长能力),按公告日(pubDate)落地为 data/fundamentals.csv

使用方(rf_model/lstm_model)在对齐到每个交易日时,必须只使用 pubDate <= 当前交易日 的最新一条记录
(前向填充),不能引入尚未公告的未来数据。
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baostock as bs
import pandas as pd

import config

PROFIT_COLS = ["roeAvg", "npMargin", "gpMargin", "netProfit", "epsTTM"]
GROWTH_COLS = ["YOYEquity", "YOYAsset", "YOYNI", "YOYEPSBasic", "YOYPNI"]


def _query_all_quarters(query_func, start_year=2018):
    """逐年逐季度调用 baostock 的季度数据接口,拼接成一个 DataFrame"""
    current_year = datetime.today().year
    current_quarter = (datetime.today().month - 1) // 3 + 1

    frames = []
    for year in range(start_year, current_year + 1):
        max_q = current_quarter if year == current_year else 4
        for quarter in range(1, max_q + 1):
            rs = query_func(code=config.STOCK_CODE, year=year, quarter=quarter)
            if rs.error_code != "0":
                continue
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                frames.append(pd.DataFrame(rows, columns=rs.fields))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_fundamentals():
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")

    try:
        profit_df = _query_all_quarters(bs.query_profit_data)
        growth_df = _query_all_quarters(bs.query_growth_data)
    finally:
        bs.logout()

    if profit_df.empty and growth_df.empty:
        raise RuntimeError("未拉取到任何财务数据")

    merged = pd.merge(
        profit_df[["code", "pubDate", "statDate"] + PROFIT_COLS] if not profit_df.empty else pd.DataFrame(columns=["code", "pubDate", "statDate"] + PROFIT_COLS),
        growth_df[["code", "pubDate", "statDate"] + GROWTH_COLS] if not growth_df.empty else pd.DataFrame(columns=["code", "pubDate", "statDate"] + GROWTH_COLS),
        on=["code", "pubDate", "statDate"],
        how="outer",
    )

    for col in PROFIT_COLS + GROWTH_COLS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.sort_values("pubDate").reset_index(drop=True)
    return merged


def save_fundamentals(df):
    df.to_csv(config.FUNDAMENTALS_CSV, index=False, encoding="utf-8-sig")


def load_fundamentals():
    return pd.read_csv(config.FUNDAMENTALS_CSV, parse_dates=["pubDate", "statDate"])


def align_to_daily(daily_dates, fundamentals_df):
    """把季度财务数据前向填充对齐到给定的交易日序列(daily_dates: 升序的日期Series/Index)

    对每个交易日 d,取 pubDate <= d 的最新一条财务记录;若d早于任何已公告数据,则为NaN。
    """
    fdf = fundamentals_df.sort_values("pubDate").reset_index(drop=True)
    dates = pd.to_datetime(pd.Series(daily_dates)).reset_index(drop=True)

    aligned = pd.merge_asof(
        pd.DataFrame({"date": dates}).sort_values("date"),
        fdf.rename(columns={"pubDate": "date"}),
        on="date",
        direction="backward",
    )
    return aligned.drop(columns=["statDate"], errors="ignore")


if __name__ == "__main__":
    config.fix_console_encoding()
    df = fetch_fundamentals()
    save_fundamentals(df)
    print(f"共拉取 {len(df)} 条季度财务记录")
    print(f"已保存到 {config.FUNDAMENTALS_CSV}")
    print(df.tail(5).to_string(index=False))
