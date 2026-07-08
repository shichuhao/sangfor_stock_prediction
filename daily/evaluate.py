"""每日任务(2/2):对 predictions.csv 中已到下一交易日、但尚未回填实际结果的记录,
用最新行情数据回填 target_date/actual_close/actual_direction/correct/abs_error。

应在 predict_today.py(已刷新行情数据)之后运行。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config
from data.fetch_data import load_history


def evaluate_pending_predictions():
    if not os.path.exists(config.PREDICTIONS_CSV):
        print("predictions.csv 不存在,无需回填")
        return None

    preds = pd.read_csv(config.PREDICTIONS_CSV, dtype=str)
    history = load_history()
    history = history.sort_values("date").reset_index(drop=True)
    trading_dates = history["date"].tolist()
    close_by_date = dict(zip(history["date"], history["close"]))

    pending_mask = preds["target_date"].isna() | (preds["target_date"] == "")
    n_pending = int(pending_mask.sum())
    if n_pending == 0:
        print("没有待回填的预测记录")
        return preds

    for idx in preds[pending_mask].index:
        issued_date = pd.Timestamp(preds.at[idx, "issued_date"])
        future_dates = [d for d in trading_dates if d > issued_date]
        if not future_dates:
            continue  # 下一交易日数据还没出来,下次再回填

        target_date = future_dates[0]
        actual_close = float(close_by_date[target_date])
        last_close = float(preds.at[idx, "last_close"])
        actual_direction = 1 if actual_close >= last_close else -1
        pred_direction = int(float(preds.at[idx, "pred_direction"]))
        pred_close = float(preds.at[idx, "pred_close"])

        # 这一列是 pandas 3.0 的严格字符串dtype(由上面 dtype=str 读入产生),
        # 直接赋值 float/int 会报 "Invalid value ... for dtype 'str'",必须显式转成 str 再赋值
        preds.at[idx, "target_date"] = target_date.strftime("%Y-%m-%d")
        preds.at[idx, "actual_close"] = str(actual_close)
        preds.at[idx, "actual_direction"] = str(actual_direction)
        preds.at[idx, "correct"] = str(int(pred_direction == actual_direction))
        preds.at[idx, "abs_error"] = str(abs(pred_close - actual_close))

    preds.to_csv(config.PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
    return preds


if __name__ == "__main__":
    config.fix_console_encoding()
    result = evaluate_pending_predictions()
    if result is not None:
        resolved = result[result["target_date"].notna() & (result["target_date"] != "")]
        print(f"累计已回填 {len(resolved)} 条预测记录,已保存到 {config.PREDICTIONS_CSV}")
