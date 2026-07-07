"""拉取情绪相关数据:
1. stock_comment_em (东方财富"千股千评"): 综合得分/机构参与度/关注指数/排名 —— 无历史查询接口,
   只能每次拉到"当前快照",所以本脚本采用【追加写入】方式,每天跑一次就往 sentiment.csv 多积累一行。
2. stock_news_em: 该股票最新新闻标题+内容,用关键词词典打分得到当日新闻情绪分。

由于千股千评没有历史接口,情绪模型无法做长历史回测,只能从本项目上线之日起自然积累每日快照。
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config

config.patch_requests_user_agent()
config.patch_pandas_for_akshare()

import akshare as ak

POSITIVE_WORDS = [
    "增长", "上涨", "预增", "中标", "合作", "签约", "大涨", "涨停", "利好", "回购",
    "增持", "扭亏", "分红", "突破", "新高", "获批", "战略合作", "超预期", "领先",
    "创新高", "订单", "中标公告", "业绩预增",
]

NEGATIVE_WORDS = [
    "下跌", "亏损", "减持", "诉讼", "处罚", "终止", "预警", "退市", "违规", "立案",
    "调查", "爆雷", "商誉减值", "下滑", "延期", "延至", "推迟", "冻结", "质押",
    "平仓", "ST", "跌停", "警示", "问询函", "关注函",
]


def fetch_comment_snapshot():
    """拉取千股千评当前快照,只保留目标股票那一行"""
    df = ak.stock_comment_em()
    sub = df[df["代码"] == config.STOCK_SYMBOL].copy()
    if sub.empty:
        return None

    row = sub.iloc[0]
    return {
        "date": str(row["交易日"]),
        "comment_score": float(row["综合得分"]),
        "institution_participation": float(row["机构参与度"]),
        "attention_index": float(row["关注指数"]),
        "current_rank": int(row["目前排名"]),
    }


def _score_text(text):
    if not isinstance(text, str):
        return 0
    pos = sum(text.count(w) for w in POSITIVE_WORDS)
    neg = sum(text.count(w) for w in NEGATIVE_WORDS)
    return pos - neg


def fetch_news_sentiment():
    """拉取最新新闻,按关键词词典打分,返回当日新闻情绪分(所有拉到的新闻打分求和)"""
    try:
        df = ak.stock_news_em(symbol=config.STOCK_SYMBOL)
    except Exception:
        return {"news_sentiment_score": 0, "news_count": 0}

    if df.empty:
        return {"news_sentiment_score": 0, "news_count": 0}

    combined_text = (df["新闻标题"].fillna("") + " " + df["新闻内容"].fillna(""))
    scores = combined_text.apply(_score_text)
    return {
        "news_sentiment_score": int(scores.sum()),
        "news_count": int(len(df)),
    }


def fetch_sentiment_snapshot():
    comment = fetch_comment_snapshot()
    news = fetch_news_sentiment()

    if comment is None:
        comment = {
            "date": datetime.today().strftime("%Y-%m-%d"),
            "comment_score": None,
            "institution_participation": None,
            "attention_index": None,
            "current_rank": None,
        }

    row = {**comment, **news}
    return row


def append_sentiment_snapshot():
    """把今天的快照追加到 sentiment.csv,若当天已存在记录则覆盖(避免同日重复运行产生多行)"""
    row = fetch_sentiment_snapshot()

    if os.path.exists(config.SENTIMENT_CSV):
        df = pd.read_csv(config.SENTIMENT_CSV)
        df = df[df["date"] != row["date"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(config.SENTIMENT_CSV, index=False, encoding="utf-8-sig")
    return row


def load_sentiment():
    return pd.read_csv(config.SENTIMENT_CSV, parse_dates=["date"])


if __name__ == "__main__":
    config.fix_console_encoding()
    row = append_sentiment_snapshot()
    print("今日情绪快照:")
    for k, v in row.items():
        print(f"  {k}: {v}")
    print(f"已追加到 {config.SENTIMENT_CSV}")
