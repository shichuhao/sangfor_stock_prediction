# 深信服(300454) 多模型股价预测 —— 准确性追踪

用多种模型预测深信服(300454)下一交易日的涨跌方向和收盘价,通过历史回测 + 每日实盘追踪两条线,
持续统计并对比各模型的预测准确性。

## 数据来源

- **行情/财务/大盘指数**: [baostock](http://baostock.com/)(免费,无需注册)
  - 日线行情(含估值指标 peTTM/pbMRQ/psTTM/pcfNcfTTM)
  - 季度财务数据(盈利能力/成长能力)
  - 行业分类
  - 创业板指(sz.399006)日线
- **情绪/新闻**: [akshare](https://akshare.akfamily.xyz/)
  - `stock_comment_em`(东方财富"千股千评"): 综合得分/机构参与度/关注指数
  - `stock_news_em`: 最新新闻标题,用关键词词典打分

> 注意: akshare 部分接口在某些网络环境下若不带浏览器 User-Agent 会被连接重置,
> 且 pandas 3.0 的 `infer_string` 默认行为会导致部分老接口的正则表达式报错。
> `config.py` 中的 `patch_requests_user_agent()` / `patch_pandas_for_akshare()` 已处理这两个问题。

## 目录结构

```
config.py                  # 公共配置(股票代码/路径/训练测试切分点)
data/
  fetch_data.py             # 日线行情(含估值扩展字段) → data/history.csv
  fetch_fundamentals.py     # 季度财务数据(按公告日对齐) → data/fundamentals.csv
  fetch_industry.py         # 行业分类 → data/industry.csv
  fetch_index.py            # 创业板指日线 → data/index.csv
  fetch_sentiment.py        # 千股千评+新闻情绪快照(逐日追加) → data/sentiment.csv
models/
  base.py                  # 统一模型接口
  naive.py                 # 基线:随机游走 / N日均线
  arima_model.py           # ARIMA
  features.py              # 多因子特征矩阵(技术面+估值面+基本面+大盘联动)
  rf_model.py              # 随机森林多因子
  lstm_model.py            # LSTM(单变量 vs 多变量对比)
  sentiment_model.py       # 情绪模型(规则式)
backtest/
  run_backtest.py          # walk-forward历史回测 → outputs/backtest_results.csv
daily/
  predict_today.py         # 刷新数据 + 对下一交易日发出预测 → outputs/predictions.csv
  evaluate.py               # 用最新行情回填此前预测的实际结果
reports/
  generate_report.py       # 汇总回测+实盘结果 → reports/accuracy_report.md
.github/workflows/
  daily_predict.yml        # 每个交易日收盘后自动跑 predict → evaluate → report → commit
  monthly_backtest.yml     # 每月1号自动用最新全部历史重跑一次完整walk-forward回测
```

## 本地运行

```bash
pip install -r requirements.txt

# 1. 拉取全部历史数据
python data/fetch_data.py
python data/fetch_fundamentals.py
python data/fetch_industry.py
python data/fetch_index.py

# 2. 历史回测(约耗时10-20分钟,视机器性能)
python backtest/run_backtest.py

# 3. 每日预测 + 回填(建议每个交易日收盘后跑一次;GitHub Actions会自动做这件事)
python daily/predict_today.py
python daily/evaluate.py

# 4. 生成报告
python reports/generate_report.py
```

## 模型说明

| 模型 | 类型 | 说明 |
|---|---|---|
| random_walk | 基线 | 预测明天收盘价=今天收盘价 |
| moving_average | 基线 | 预测明天收盘价=最近5日均价 |
| arima | 时间序列 | 仅用收盘价序列,statsmodels ARIMA(1,1,1) |
| random_forest | 多因子机器学习 | 技术面+估值面+基本面+大盘联动因子,随机森林分类+回归 |
| lstm_univariate | 深度学习 | 仅收盘价序列的LSTM,用于和多变量版对比 |
| lstm_multivariate | 深度学习 | 收盘价+成交量+换手率+PE+PB+ROE+净利润同比增速的多变量LSTM |
| sentiment | 规则式 | 千股千评综合得分动量 + 新闻情绪分,方向预测为主 |

## 已知局限性(如实说明,不夸大模型效果)

1. **股价随机性**: 即便数据量充足,方向准确率也很难显著超过50-55%,报告如实呈现回测结果。
2. **情绪模型无法历史回测**: `stock_comment_em` 没有历史查询接口,只能从项目上线之日起逐日积累快照,
   因此该模型不参与 `run_backtest.py` 的历史回测,只能通过每日实盘追踪评估,初期样本量会很小。
3. **情绪模型的价格误差是占位值**: 该模型核心产出是方向信号,pred_close 是按固定假设幅度(1%)
   外推得到的,其 MAE/RMSE 不应与其它模型直接比较。
4. **LSTM 非每日重训**: 为控制训练耗时,回测/生产环境中每10个交易日才重新训练一次LSTM权重,
   期间用最新数据窗口做推理,这也更贴近真实生产环境"定期重训+每日推理"的做法。
5. **实盘追踪样本随时间积累**: 需要 GitHub Actions 连续运行数周才能积累到有统计意义的样本量,
   项目上线初期该部分结论的置信度较低。
6. **历史回测本身不是每天重跑的**: `backtest/run_backtest.py` 完整跑一遍约耗时10-20分钟,不适合放进
   每日流程。`reports/accuracy_report.md` 第2节"历史回测结果"是按月刷新的静态基准(见下方 GitHub
   Actions 说明);报告第1节"累计评估结果"把这个静态基准和每天新增的实盘追踪记录拼接起来,
   是唯一会随每天运行持续增长、覆盖最新交易日的整体指标。

## GitHub Actions

- `.github/workflows/daily_predict.yml`: 每个工作日北京时间16:35(A股收盘后)自动
  拉取最新数据 → 对下一交易日发出预测 → 回填上一次预测的实际结果 → 重新生成报告 → 提交回仓库。
- `.github/workflows/monthly_backtest.yml`: 每月1号自动用最新全部历史重新训练并跑一遍完整的
  walk-forward历史回测,刷新报告第2节的静态基准表。

两者都使用默认的 `GITHUB_TOKEN`,无需额外配置密钥。
