"""所有模型的统一接口"""


class BaseModel:
    name = "base"
    # 回测/每日追踪时,每隔多少个交易日重新fit一次(1=每天都重新拟合)。
    # 训练成本高的模型(如LSTM)可以调大这个值,期间仍用最新数据做推理,只是权重不每天更新,
    # 这也更贴近真实生产环境中"定期重训+每日推理"的做法。
    retrain_interval = 1

    def fit(self, train_df):
        """用 train_df(按日期升序、截止到某一天的历史数据)训练/拟合模型"""
        raise NotImplementedError

    def predict_next(self, recent_df):
        """基于 recent_df(截止到当前的历史,通常与训练时的数据一致或更新)预测下一交易日

        返回 dict: {"pred_close": float, "pred_direction": 1 | -1, "pred_pct_change": float | None}
        pred_direction: 1=预测涨, -1=预测跌(不预测持平,视为跌以便二分类评估)
        """
        raise NotImplementedError

    @staticmethod
    def direction_from_prices(last_close, pred_close):
        return 1 if pred_close >= last_close else -1
