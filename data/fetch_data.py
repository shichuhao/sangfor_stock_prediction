"""拉取/更新股票日线行情(含估值扩展字段),落地为 data/history.csv"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baostock as bs
import pandas as pd
from datetime import datetime

import config

FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)

RENAME_MAP = {
    "date": "date",
    "code": "code",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "preclose": "preclose",
    "volume": "volume",
    "amount": "amount",
    "turn": "turnover_rate",
    "tradestatus": "tradestatus",
    "pctChg": "pct_change",
    "peTTM": "pe_ttm",
    "pbMRQ": "pb_mrq",
    "psTTM": "ps_ttm",
    "pcfNcfTTM": "pcf_ncf_ttm",
    "isST": "is_st",
}

NUMERIC_COLS = [
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "turnover_rate", "pct_change", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ncf_ttm",
]


def fetch_history(end_date=None):
    """从 baostock 拉取该股票全部可得历史日线(前复权),返回整理后的 DataFrame"""
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")

    try:
        rs = bs.query_history_k_data_plus(
            config.STOCK_CODE,
            FIELDS,
            start_date=config.HISTORY_START_DATE,
            end_date=end_date,
            frequency="d",
            adjustflag="2",  # 前复权
        )
        if rs.error_code != "0":
            raise RuntimeError(f"查询历史行情失败: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()

    df = df.rename(columns=RENAME_MAP)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce").astype("Int64")
    df["is_st"] = pd.to_numeric(df["is_st"], errors="coerce").astype("Int64")

    # 只保留正常交易日(tradestatus=1),停牌日的行情字段无意义
    df = df[df["tradestatus"] == 1].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_history(df):
    df.to_csv(config.HISTORY_CSV, index=False, encoding="utf-8-sig")


def load_history():
    df = pd.read_csv(config.HISTORY_CSV, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    config.fix_console_encoding()
    df = fetch_history()
    save_history(df)
    print(f"共拉取 {len(df)} 条交易日数据,时间范围 {df['date'].min()} ~ {df['date'].max()}")
    print(f"已保存到 {config.HISTORY_CSV}")
    print(df.tail(5).to_string(index=False))
