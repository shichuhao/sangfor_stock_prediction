"""拉取行业分类(静态信息),落地为 data/industry.csv"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baostock as bs
import pandas as pd

import config


def fetch_industry():
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")

    try:
        rs = bs.query_stock_industry(code=config.STOCK_CODE)
        if rs.error_code != "0":
            raise RuntimeError(f"查询行业分类失败: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()

    return df


def save_industry(df):
    df.to_csv(config.INDUSTRY_CSV, index=False, encoding="utf-8-sig")


def load_industry():
    return pd.read_csv(config.INDUSTRY_CSV)


if __name__ == "__main__":
    config.fix_console_encoding()
    df = fetch_industry()
    save_industry(df)
    print(f"已保存到 {config.INDUSTRY_CSV}")
    print(df.to_string(index=False))
