"""拉取创业板指(基准指数)日线,用于计算个股相对大盘的超额收益/beta因子,落地为 data/index.csv"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baostock as bs
import pandas as pd
from datetime import datetime

import config


def fetch_index(end_date=None):
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")

    try:
        rs = bs.query_history_k_data_plus(
            config.INDEX_CODE,
            "date,code,close,pctChg",
            start_date=config.HISTORY_START_DATE,
            end_date=end_date,
            frequency="d",
        )
        if rs.error_code != "0":
            raise RuntimeError(f"查询指数行情失败: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()

    df = df.rename(columns={"close": "index_close", "pctChg": "index_pct_change"})
    df["index_close"] = pd.to_numeric(df["index_close"], errors="coerce")
    df["index_pct_change"] = pd.to_numeric(df["index_pct_change"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_index(df):
    df.to_csv(config.INDEX_CSV, index=False, encoding="utf-8-sig")


def load_index():
    return pd.read_csv(config.INDEX_CSV, parse_dates=["date"])


if __name__ == "__main__":
    config.fix_console_encoding()
    df = fetch_index()
    save_index(df)
    print(f"共拉取 {len(df)} 条指数数据,时间范围 {df['date'].min()} ~ {df['date'].max()}")
    print(f"已保存到 {config.INDEX_CSV}")
    print(df.tail(3).to_string(index=False))
