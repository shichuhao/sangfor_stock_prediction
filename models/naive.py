"""简单基线模型:随机游走 + N日移动平均"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import BaseModel


class RandomWalkModel(BaseModel):
    """预测明天收盘价 = 今天收盘价(即预测不涨不跌,方向按持平记为跌处理)"""

    name = "random_walk"

    def fit(self, train_df):
        return self

    def predict_next(self, recent_df):
        last_close = float(recent_df["close"].iloc[-1])
        return {
            "pred_close": last_close,
            "pred_direction": -1,  # 随机游走本身不预测方向,统一记为-1(与"涨"区分开)
            "pred_pct_change": 0.0,
        }


class MovingAverageModel(BaseModel):
    """预测明天收盘价 = 最近N日收盘价均值"""

    name = "moving_average"

    def __init__(self, window=5):
        self.window = window

    def fit(self, train_df):
        return self

    def predict_next(self, recent_df):
        last_close = float(recent_df["close"].iloc[-1])
        window_data = recent_df["close"].iloc[-self.window:]
        pred_close = float(window_data.mean())
        direction = self.direction_from_prices(last_close, pred_close)
        pred_pct_change = (pred_close - last_close) / last_close * 100
        return {
            "pred_close": pred_close,
            "pred_direction": direction,
            "pred_pct_change": pred_pct_change,
        }
