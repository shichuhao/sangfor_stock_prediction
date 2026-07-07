"""情绪模型:基于千股千评综合得分动量 + 新闻情绪分的规则式方向预测

重要限制(已在README/报告中注明):
1. stock_comment_em 没有历史查询接口,sentiment.csv 只能从项目上线之日起逐日积累快照,
   因此本模型【无法参与历史 walk-forward 回测】,只能通过每日实盘追踪来评估。
2. 本模型是规则式模型(非机器学习训练),核心产出是"方向"信号;由于情绪信号本身不携带
   价格变动幅度的信息,pred_close 是用固定的 ASSUMED_MOVE_PCT(默认1%)按预测方向外推得到的
   占位值,仅用于满足"价格误差(MAE/RMSE)"这个统一评估口径,不代表该模型真的具备价格幅度预测能力,
   其 MAE/RMSE 数值在报告中会与其它模型分开解读,不直接比较。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from models.base import BaseModel

ASSUMED_MOVE_PCT = 1.0


class SentimentModel(BaseModel):
    name = "sentiment"
    retrain_interval = 1  # 规则式模型无需训练,fit()是空操作

    def fit(self, train_df):
        return self

    def predict_next(self, recent_df):
        last_close = float(recent_df["close"].iloc[-1])

        score_momentum = 0
        if "comment_score" in recent_df.columns and len(recent_df) >= 2:
            s = recent_df["comment_score"]
            if pd.notna(s.iloc[-1]) and pd.notna(s.iloc[-2]):
                score_momentum = s.iloc[-1] - s.iloc[-2]

        news_score = 0
        if "news_sentiment_score" in recent_df.columns:
            v = recent_df["news_sentiment_score"].iloc[-1]
            news_score = 0 if pd.isna(v) else v

        if score_momentum > 0 and news_score > 0:
            direction = 1
        elif score_momentum < 0 and news_score < 0:
            direction = -1
        elif score_momentum != 0:
            direction = 1 if score_momentum > 0 else -1
        elif news_score != 0:
            direction = 1 if news_score > 0 else -1
        else:
            direction = -1  # 无任何信号时,保守预测为跌(与随机游走基线一致的默认假设)

        pred_pct_change = ASSUMED_MOVE_PCT * direction
        pred_close = last_close * (1 + pred_pct_change / 100)

        return {
            "pred_close": pred_close,
            "pred_direction": direction,
            "pred_pct_change": pred_pct_change,
        }


def build_price_sentiment_frame(history_df, sentiment_df):
    """把行情表(需含date,close)和情绪快照表按date左连接,供SentimentModel使用"""
    merged = history_df[["date", "close"]].merge(sentiment_df, on="date", how="left")
    return merged.sort_values("date").reset_index(drop=True)
