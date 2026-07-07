# 改动记录

## 2026-07-07 项目初始化

从零搭建"深信服(300454)多模型预测准确性追踪"项目,目标是对比多种股价预测方法的实际准确率,
并通过历史回测+每日实盘追踪两条线持续积累证据,而不是只做一次性的模型演示。

### 数据层
- `data/fetch_data.py`: 用 baostock 拉取全部历史日线行情(2018-05-16上市至今,共1976个交易日),
  含估值扩展字段(peTTM/pbMRQ/psTTM/pcfNcfTTM)。
- `data/fetch_fundamentals.py`: 拉取季度盈利能力(ROE/净利率/毛利率/EPS)和成长能力
  (净资产/净利润/EPS同比增速)数据,按公告日(pubDate)前向对齐到每个交易日,避免引入未来数据。
- `data/fetch_industry.py`: 拉取行业分类(静态信息)。
- `data/fetch_index.py`: 拉取创业板指(sz.399006)日线,用于计算个股相对大盘的超额收益/beta因子。
- `data/fetch_sentiment.py`: 拉取东方财富"千股千评"(综合得分/机构参与度/关注指数)+ 个股新闻标题,
  用关键词词典给新闻打情绪分。因该接口无历史查询能力,采用逐日追加快照的方式积累历史。

**踩坑记录**:
- akshare 部分接口(如 `stock_news_em`、`stock_comment_em`)在本机网络环境下若不带浏览器
  User-Agent 会被连接重置(`RemoteDisconnected`),已在 `config.patch_requests_user_agent()` 中
  统一打补丁解决。
- pandas 3.0 默认的 `future.infer_string` 会导致 akshare 部分老接口内部的正则表达式抛
  `ArrowInvalid: invalid escape sequence`,已在 `config.patch_pandas_for_akshare()` 中关闭该选项解决。

### 模型层(5个模型体系)
- `models/naive.py`: 随机游走 + 5日均线两个基线模型。
- `models/arima_model.py`: ARIMA(1,1,1),仅用收盘价序列。
- `models/features.py`: 构造技术面(动量/波动率/量能/均线偏离)+ 估值面(PE/PB/PS/PCF及历史分位)
  + 基本面(ROE/净利率/毛利率/EPS/同比增速)+ 大盘联动(超额收益/beta)四类因子的统一特征矩阵,
  供随机森林和LSTM共用。
- `models/rf_model.py`: 随机森林多因子模型(分类预测方向 + 回归预测涨跌幅),训练后输出特征重要性。
- `models/lstm_model.py`: 基于 PyTorch 实现,提供单变量(仅收盘价)和多变量(收盘价+成交量+换手率+
  PE+PB+ROE+净利润同比增速)两个版本,用于量化回答"加入估值/基本面因子是否真的提升了准确率"。
  为控制训练耗时,回测时每10个交易日重新训练一次(而非每天重训)。
- `models/sentiment_model.py`: 规则式模型,基于千股千评综合得分动量+新闻情绪分方向预测涨跌。
  因数据源无历史查询接口,不参与历史回测,只能通过每日实盘追踪评估。

### 评估层
- `backtest/run_backtest.py`: 对基线/ARIMA/随机森林/LSTM(单变量+多变量)做2025-01-01至今
  (约363个交易日)的walk-forward历史回测,输出方向准确率+MAE+RMSE对比。
- `daily/predict_today.py` + `daily/evaluate.py`: 每日刷新数据、对下一交易日发出7个模型
  (含情绪模型)的预测、并用次日实际收盘价回填此前预测结果,写入 `outputs/predictions.csv`。
- `reports/generate_report.py`: 汇总回测结果+最近30天实盘追踪结果+随机森林特征重要性,
  生成 `reports/accuracy_report.md`。

### 自动化
- `.github/workflows/daily_predict.yml`: 每个工作日北京时间16:35(A股收盘后)自动运行
  predict → evaluate → 重新生成报告 → 提交回仓库,使用默认 `GITHUB_TOKEN`。

### 已知局限性(详见 README.md)
股价预测的方向准确率理论上很难显著超过50-55%;情绪模型无历史回测、其价格误差为占位值;
LSTM非每日重训;实盘追踪样本需要时间自然积累。以上均未在报告中做美化处理。
