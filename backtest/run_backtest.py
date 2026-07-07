"""历史 walk-forward 回测:用 TRAIN_TEST_SPLIT_DATE 之后的每个交易日做滚动预测评估

对每个测试日 i: 只用 i 之前(不含i)的数据训练/拟合模型,用截至 i(含)的数据做特征输入,
预测第 i+1 日的收盘价/涨跌方向,与实际值比较。retrain_interval > 1 的模型(如LSTM)不每天重训,
按间隔周期重训,期间用最新数据推理,更贴近真实生产环境的做法。

情绪模型(sentiment_model)不参与本回测:stock_comment_em无历史查询接口,无法回溯历史情绪快照。
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config
from models.features import build_feature_frame
from models.naive import RandomWalkModel, MovingAverageModel
from models.arima_model import ArimaModel
from models.rf_model import RandomForestModel
from models.lstm_model import make_multivariate_model, make_univariate_model


def get_models():
    return [
        RandomWalkModel(),
        MovingAverageModel(window=5),
        ArimaModel(),
        RandomForestModel(),
        make_univariate_model(),
        make_multivariate_model(),
    ]


def run_backtest(progress_every=20):
    df = build_feature_frame().reset_index(drop=True)

    test_mask = (df["date"] >= pd.Timestamp(config.TRAIN_TEST_SPLIT_DATE)) & df["next_close"].notna()
    test_indices = df.index[test_mask].tolist()

    if not test_indices:
        raise RuntimeError("没有可用的回测样本,请检查 TRAIN_TEST_SPLIT_DATE 设置")

    print(
        f"回测区间: {df.loc[test_indices[0], 'date'].date()} ~ "
        f"{df.loc[test_indices[-1], 'date'].date()}, 共 {len(test_indices)} 个交易日"
    )

    models = get_models()
    fitted_state = {m.name: {"step": -1} for m in models}
    results = []

    t_start = time.time()
    for step_idx, i in enumerate(test_indices):
        train_df = df.iloc[:i]
        recent_df = df.iloc[: i + 1]
        row = df.iloc[i]

        for m in models:
            state = fitted_state[m.name]
            need_fit = state["step"] == -1 or (step_idx - state["step"]) >= m.retrain_interval
            if need_fit:
                try:
                    m.fit(train_df)
                    state["step"] = step_idx
                except Exception as e:
                    print(f"[警告] {m.name} 在 {row['date'].date()} fit失败: {e}")
                    continue

            try:
                pred = m.predict_next(recent_df)
            except Exception as e:
                print(f"[警告] {m.name} 在 {row['date'].date()} predict失败: {e}")
                continue

            actual_close = float(row["next_close"])
            actual_direction = int(row["next_direction"])
            correct = int(pred["pred_direction"] == actual_direction)
            abs_error = abs(pred["pred_close"] - actual_close)

            results.append(
                {
                    "date": row["date"],
                    "model": m.name,
                    "pred_close": pred["pred_close"],
                    "actual_close": actual_close,
                    "pred_direction": pred["pred_direction"],
                    "actual_direction": actual_direction,
                    "correct": correct,
                    "abs_error": abs_error,
                }
            )

        if (step_idx + 1) % progress_every == 0 or step_idx == len(test_indices) - 1:
            elapsed = time.time() - t_start
            print(f"进度 {step_idx + 1}/{len(test_indices)}, 累计耗时 {elapsed:.1f}s")

    return pd.DataFrame(results)


def save_results(df):
    df.to_csv(config.BACKTEST_RESULTS_CSV, index=False, encoding="utf-8-sig")


def summarize(df):
    summary = df.groupby("model").agg(
        n=("correct", "size"),
        direction_accuracy=("correct", "mean"),
        mae=("abs_error", "mean"),
    )
    summary["rmse"] = df.groupby("model")["abs_error"].apply(lambda e: (e**2).mean() ** 0.5)
    return summary.sort_values("direction_accuracy", ascending=False)


if __name__ == "__main__":
    config.fix_console_encoding()
    result_df = run_backtest()
    save_results(result_df)
    print(f"\n回测完成,共 {len(result_df)} 条记录,已保存到 {config.BACKTEST_RESULTS_CSV}\n")
    print(summarize(result_df).to_string())
