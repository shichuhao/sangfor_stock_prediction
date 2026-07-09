"""汇总历史回测结果 + 每日实盘追踪结果,生成 accuracy_report.md"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config

MODEL_DISPLAY_NAMES = {
    "random_walk": "随机游走(基线)",
    "moving_average": "5日均线(基线)",
    "arima": "ARIMA",
    "random_forest": "随机森林多因子",
    "lstm_univariate": "LSTM(仅价格,单变量)",
    "lstm_multivariate": "LSTM(价格+估值+基本面,多变量)",
    "sentiment": "情绪模型(千股千评+新闻)",
}


def summarize_direction_and_error(df, group_col="model"):
    summary = df.groupby(group_col).agg(
        样本数=("correct", "size"),
        方向准确率=("correct", "mean"),
        MAE=("abs_error", "mean"),
    )
    summary["RMSE"] = df.groupby(group_col)["abs_error"].apply(lambda e: (e**2).mean() ** 0.5)
    summary["方向准确率"] = (summary["方向准确率"] * 100).round(2)
    summary["MAE"] = summary["MAE"].round(4)
    summary["RMSE"] = summary["RMSE"].round(4)
    return summary.sort_values("方向准确率", ascending=False)


def load_backtest_summary():
    if not os.path.exists(config.BACKTEST_RESULTS_CSV):
        return None, None
    df = pd.read_csv(config.BACKTEST_RESULTS_CSV, parse_dates=["date"])
    summary = summarize_direction_and_error(df)
    date_range = (df["date"].min().date(), df["date"].max().date())
    return summary, date_range


def _load_resolved_predictions():
    """加载 predictions.csv 中已回填实际结果的记录,统一数值类型"""
    if not os.path.exists(config.PREDICTIONS_CSV):
        return pd.DataFrame()
    df = pd.read_csv(config.PREDICTIONS_CSV)
    df = df[df["target_date"].notna() & (df["target_date"] != "")].copy()
    if df.empty:
        return pd.DataFrame()

    for col in ["correct", "abs_error", "pred_direction", "actual_direction", "pred_close", "actual_close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["target_date"])
    return df


def load_live_tracking_summary(window_days=30):
    df = _load_resolved_predictions()
    if df.empty:
        return None, 0

    cutoff = datetime.today() - timedelta(days=window_days)
    recent = df[df["date"] >= cutoff]
    if recent.empty:
        return None, 0

    summary = summarize_direction_and_error(recent)
    return summary, len(recent["date"].unique())


def load_combined_summary():
    """历史回测(定期刷新的静态基准) + 每日实盘追踪(逐日增长) 拼接成一条连续时间线,
    两者本身首尾相接(回测截止日之后由实盘追踪接续),拼接后无需重新跑一遍耗时的历史回测
    就能得到一个随每日运行持续增长、覆盖到最新交易日的整体评估结果。
    """
    frames = []
    if os.path.exists(config.BACKTEST_RESULTS_CSV):
        frames.append(pd.read_csv(config.BACKTEST_RESULTS_CSV, parse_dates=["date"]))

    live_df = _load_resolved_predictions()
    if not live_df.empty:
        cols = ["date", "model", "pred_close", "actual_close", "pred_direction", "actual_direction", "correct", "abs_error"]
        frames.append(live_df[cols])

    if not frames:
        return None, None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "model"], keep="last")
    combined = combined.sort_values("date")

    summary = summarize_direction_and_error(combined)
    date_range = (combined["date"].min().date(), combined["date"].max().date())
    return summary, (date_range, len(combined))


def get_feature_importance_top(n=8):
    try:
        from models.features import build_feature_frame, ALL_FEATURES
        from models.rf_model import RandomForestModel

        df = build_feature_frame().dropna(subset=ALL_FEATURES + ["next_direction"])
        m = RandomForestModel()
        m.fit(df.iloc[:-1])
        items = sorted(m.feature_importances_.items(), key=lambda x: -x[1])[:n]
        return items
    except Exception as e:
        return [("获取特征重要性失败", str(e))]


def render_table(df):
    df = df.rename(index=MODEL_DISPLAY_NAMES)
    return df.to_markdown()


def generate_report():
    lines = []
    lines.append(f"# 深信服(300454) 多模型预测准确性报告\n")
    lines.append(f"生成时间: {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 1. 累计评估结果(历史回测+实盘追踪拼接,随每日运行持续增长)\n")
    combined_summary, combined_meta = load_combined_summary()
    if combined_summary is not None:
        (c_start, c_end), n_records = combined_meta
        lines.append(f"覆盖区间: {c_start} ~ {c_end}(共 {n_records} 条评估记录)\n")
        lines.append(render_table(combined_summary) + "\n")
        lines.append(
            "\n说明: 这是历史回测(截止到最近一次月度回测的日期)和每日实盘追踪(此后每个交易日)"
            "拼接后的整体结果,不需要重新跑一遍完整回测就能反映最新数据,是本报告里唯一会随时间"
            "持续变化的整体指标。\n"
        )
    else:
        lines.append("尚无可用数据,请先执行 `python backtest/run_backtest.py`。\n")

    lines.append("\n## 2. 历史回测结果(walk-forward,静态基准,按月刷新)\n")
    bt_summary, bt_range = load_backtest_summary()
    if bt_summary is not None:
        lines.append(f"回测区间: {bt_range[0]} ~ {bt_range[1]}\n")
        lines.append(render_table(bt_summary) + "\n")
        lines.append(
            "\n说明: 该表由 `.github/workflows/monthly_backtest.yml` 每月自动重跑一次(用最新全部历史"
            "重新训练),平时(每日运行)不会变化;情绪模型不参与历史回测(数据源无历史查询接口),"
            "以上仅覆盖基线/ARIMA/随机森林/LSTM四类可回测模型。若想看每天都在增长的整体数字,请看第1节。\n"
        )
    else:
        lines.append("尚未运行回测,请先执行 `python backtest/run_backtest.py`。\n")

    lines.append("\n## 3. 每日实盘追踪结果(最近30天)\n")
    live_summary, n_days = load_live_tracking_summary(window_days=30)
    if live_summary is not None:
        lines.append(f"覆盖交易日数: {n_days}\n")
        lines.append(render_table(live_summary) + "\n")
    else:
        lines.append(
            "暂无可用的实盘追踪样本(需GitHub Actions连续运行数日、累积到有实际结果回填的预测后才有数据)。\n"
        )

    lines.append("\n## 4. 随机森林多因子模型 — 特征重要性 Top8\n")
    for name, importance in get_feature_importance_top():
        if isinstance(importance, float):
            lines.append(f"- {name}: {importance:.4f}")
        else:
            lines.append(f"- {name}: {importance}")

    lines.append("\n## 5. 局限性说明\n")
    lines.append("- 股价预测本身受市场随机性影响很大,即便数据量充足,方向准确率也很难显著超过50-55%,"
                 "以上结果如实呈现,不代表可直接用于实盘交易决策。")
    lines.append("- 情绪模型的 MAE/RMSE 是按固定假设幅度(1%)外推得到的占位值,"
                 "仅用于满足统一评估口径,不代表该模型具备价格幅度预测能力,不应与其它模型的价格误差直接比较。")
    lines.append("- LSTM 模型为控制回测耗时,采用每10个交易日重新训练一次(而非每天重训),"
                 "期间使用最新窗口数据做推理,更贴近真实生产环境的做法。")
    lines.append("- 每日实盘追踪需要 GitHub Actions 连续运行才能积累样本,项目上线初期该部分样本量会很小,"
                 "结论置信度随运行天数增加而提升。")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    config.fix_console_encoding()
    report = generate_report()
    with open(config.ACCURACY_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n已保存到 {config.ACCURACY_REPORT_MD}")
