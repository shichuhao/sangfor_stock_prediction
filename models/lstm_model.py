"""LSTM 序列模型:提供多变量版(价格+成交量+估值+基本面)和单变量版(仅收盘价)两个实例,
用于对比"加入估值/基本面因子是否真的提升预测准确率"。

与 rf_model 一样,fit()/predict_next() 接收 models.features.build_feature_frame() 产出的
已因子化 DataFrame。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from models.base import BaseModel

MULTIVARIATE_FEATURES = ["close", "volume", "turnover_rate", "pe_ttm", "pb_mrq", "roeAvg", "YOYNI"]
UNIVARIATE_FEATURES = ["close"]


class _LSTMNet:
    """延迟导入torch,避免在未安装torch的环境里import本模块就报错"""

    def __new__(cls, input_size, hidden_size=32, num_layers=1):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        return Net()


class LSTMModel(BaseModel):
    # 训练成本较高(单次fit约数秒),回测时每隔10个交易日重训一次,期间用最新窗口数据做推理
    retrain_interval = 10

    def __init__(self, feature_cols, window=30, hidden_size=32, epochs=60, lr=1e-3, name="lstm"):
        self.feature_cols = feature_cols
        self.window = window
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.name = name
        self.scaler = None
        self.net = None

    def _make_windows(self, arr, targets):
        X, y = [], []
        for i in range(self.window, len(arr)):
            X.append(arr[i - self.window:i])
            y.append(targets[i])
        return np.stack(X), np.array(y)

    def fit(self, train_df):
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        df = train_df.dropna(subset=self.feature_cols + ["next_pct_change"]).reset_index(drop=True)
        if len(df) <= self.window + 10:
            raise ValueError(f"{self.name}: 可用训练样本({len(df)})不足以构造窗口长度{self.window}的序列")

        feats = df[self.feature_cols].values.astype(np.float32)
        self.scaler = StandardScaler().fit(feats)
        feats_scaled = self.scaler.transform(feats)
        targets = df["next_pct_change"].values.astype(np.float32)

        X, y = self._make_windows(feats_scaled, targets)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        torch.manual_seed(config.RANDOM_SEED)
        self.net = _LSTMNet(input_size=len(self.feature_cols), hidden_size=self.hidden_size)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.net(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict_next(self, recent_df):
        import torch

        df = recent_df.dropna(subset=self.feature_cols).reset_index(drop=True)
        last_close = float(df["close"].iloc[-1])
        window_raw = df[self.feature_cols].values.astype(np.float32)[-self.window:]

        if len(window_raw) < self.window:
            pad = np.repeat(window_raw[:1], self.window - len(window_raw), axis=0)
            window_raw = np.concatenate([pad, window_raw], axis=0)

        window_scaled = self.scaler.transform(window_raw)
        X = torch.tensor(window_scaled[np.newaxis, :, :], dtype=torch.float32)

        self.net.eval()
        with torch.no_grad():
            pred_pct_change = float(self.net(X).item())

        pred_close = last_close * (1 + pred_pct_change / 100)
        direction = self.direction_from_prices(last_close, pred_close)
        return {
            "pred_close": pred_close,
            "pred_direction": direction,
            "pred_pct_change": pred_pct_change,
        }


def make_multivariate_model():
    return LSTMModel(MULTIVARIATE_FEATURES, name="lstm_multivariate")


def make_univariate_model():
    return LSTMModel(UNIVARIATE_FEATURES, name="lstm_univariate")
