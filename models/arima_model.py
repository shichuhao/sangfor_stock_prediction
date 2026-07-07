"""ARIMA 时间序列模型:仅用历史收盘价序列做外推"""

import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from models.base import BaseModel

warnings.filterwarnings("ignore")


class ArimaModel(BaseModel):
    name = "arima"

    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self._last_close = None

    def fit(self, train_df):
        self._last_close = float(train_df["close"].iloc[-1])
        return self

    def predict_next(self, recent_df):
        from statsmodels.tsa.arima.model import ARIMA

        close_series = recent_df["close"].astype(float).reset_index(drop=True)
        last_close = float(close_series.iloc[-1])

        try:
            model = ARIMA(close_series, order=self.order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)
            pred_close = float(np.asarray(forecast)[0])
            if not np.isfinite(pred_close):
                raise ValueError("ARIMA 预测结果非有限值")
        except Exception:
            # 拟合失败(如数据太短/不收敛)时退化为随机游走,保证流程不中断
            pred_close = last_close

        direction = self.direction_from_prices(last_close, pred_close)
        pred_pct_change = (pred_close - last_close) / last_close * 100
        return {
            "pred_close": pred_close,
            "pred_direction": direction,
            "pred_pct_change": pred_pct_change,
        }
