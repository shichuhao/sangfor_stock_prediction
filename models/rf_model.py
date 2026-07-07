"""随机森林多因子模型:技术面+估值面+基本面+大盘联动

注意: 与 naive/arima 不同,本模型的 fit()/predict_next() 接收的是
models.features.build_feature_frame() 产出的"已因子化"DataFrame,而不是原始行情表。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.base import BaseModel
from models.features import ALL_FEATURES


class RandomForestModel(BaseModel):
    name = "random_forest"

    def __init__(self, n_estimators=300, max_depth=6):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.clf = None
        self.reg = None
        self.feature_importances_ = None

    def fit(self, train_df):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        usable = train_df.dropna(subset=ALL_FEATURES + ["next_direction", "next_pct_change"])
        X = usable[ALL_FEATURES]
        y_dir = usable["next_direction"]
        y_pct = usable["next_pct_change"]

        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ).fit(X, y_dir)

        self.reg = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ).fit(X, y_pct)

        self.feature_importances_ = dict(zip(ALL_FEATURES, self.clf.feature_importances_))
        return self

    def predict_next(self, recent_df):
        last_row = recent_df.iloc[-1]
        last_close = float(last_row["close"])
        X_last = last_row[ALL_FEATURES].to_frame().T

        pred_direction = int(self.clf.predict(X_last)[0])
        pred_pct_change = float(self.reg.predict(X_last)[0])
        pred_close = last_close * (1 + pred_pct_change / 100)

        return {
            "pred_close": pred_close,
            "pred_direction": pred_direction,
            "pred_pct_change": pred_pct_change,
        }
