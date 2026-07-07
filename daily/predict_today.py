"""每日任务(1/2):刷新数据 + 用截至当前的全部历史,对每个模型预测下一交易日,追加写入 predictions.csv

设计:每条预测记录先只包含 issued_date(发出预测所用的最新数据日期)、last_close(当天收盘价,
用于后续判断实际涨跌方向)、模型的预测值;target_date及实际结果由 evaluate.py 在下一次交易日
数据到位后回填。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config
from data.fetch_data import fetch_history, save_history
from data.fetch_fundamentals import fetch_fundamentals, save_fundamentals
from data.fetch_industry import fetch_industry, save_industry
from data.fetch_index import fetch_index, save_index
from data.fetch_sentiment import append_sentiment_snapshot, load_sentiment
from models.features import build_feature_frame
from models.naive import RandomWalkModel, MovingAverageModel
from models.arima_model import ArimaModel
from models.rf_model import RandomForestModel
from models.lstm_model import make_multivariate_model, make_univariate_model
from models.sentiment_model import SentimentModel, build_price_sentiment_frame


def refresh_all_data():
    print("刷新行情数据...")
    save_history(fetch_history())

    print("刷新财务数据...")
    save_fundamentals(fetch_fundamentals())

    print("刷新行业分类...")
    save_industry(fetch_industry())

    print("刷新大盘指数...")
    save_index(fetch_index())

    print("刷新情绪快照...")
    sentiment_row = append_sentiment_snapshot()
    print(f"  今日情绪快照: {sentiment_row}")


def get_backtestable_models():
    """有价格因子/训练能力的模型,用 features 数据框驱动"""
    return [
        RandomWalkModel(),
        MovingAverageModel(window=5),
        ArimaModel(),
        RandomForestModel(),
        make_univariate_model(),
        make_multivariate_model(),
    ]


def predict_today():
    feat_df = build_feature_frame().reset_index(drop=True)
    issued_date = feat_df["date"].iloc[-1]
    last_close = float(feat_df["close"].iloc[-1])

    predictions = []

    # ---------- 价格类模型:用全部历史(不含最后一行的未知标签)训练,推理用截至今天的数据 ----------
    train_df = feat_df.iloc[:-1]
    recent_df = feat_df

    for m in get_backtestable_models():
        try:
            m.fit(train_df)
            pred = m.predict_next(recent_df)
        except Exception as e:
            print(f"[警告] {m.name} 预测失败: {e}")
            continue
        predictions.append({"model": m.name, **pred})

    # ---------- 情绪模型:用价格+情绪快照合并表 ----------
    try:
        sentiment_df = load_sentiment()
        price_sentiment_df = build_price_sentiment_frame(feat_df[["date", "close"]], sentiment_df)
        sm = SentimentModel()
        sm.fit(price_sentiment_df)
        pred = sm.predict_next(price_sentiment_df)
        predictions.append({"model": sm.name, **pred})
    except Exception as e:
        print(f"[警告] sentiment 预测失败: {e}")

    rows = [
        {
            "issued_date": issued_date.strftime("%Y-%m-%d"),
            "last_close": last_close,
            "model": p["model"],
            "pred_close": p["pred_close"],
            "pred_direction": p["pred_direction"],
            "pred_pct_change": p["pred_pct_change"],
            "target_date": "",
            "actual_close": "",
            "actual_direction": "",
            "correct": "",
            "abs_error": "",
        }
        for p in predictions
    ]
    return pd.DataFrame(rows)


def append_predictions(new_rows_df):
    if os.path.exists(config.PREDICTIONS_CSV):
        existing = pd.read_csv(config.PREDICTIONS_CSV, dtype=str)
        # 避免同一天对同一模型重复发出预测(同日重复运行覆盖旧记录)
        key = existing["issued_date"] + "|" + existing["model"]
        new_key = new_rows_df["issued_date"].astype(str) + "|" + new_rows_df["model"].astype(str)
        existing = existing[~key.isin(new_key)]
        combined = pd.concat([existing, new_rows_df.astype(str)], ignore_index=True)
    else:
        combined = new_rows_df.astype(str)

    combined.to_csv(config.PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
    return combined


if __name__ == "__main__":
    config.fix_console_encoding()
    refresh_all_data()
    new_rows = predict_today()
    print("\n今日各模型对下一交易日的预测:")
    print(new_rows[["model", "pred_close", "pred_direction", "pred_pct_change"]].to_string(index=False))
    append_predictions(new_rows)
    print(f"\n已追加到 {config.PREDICTIONS_CSV}")
