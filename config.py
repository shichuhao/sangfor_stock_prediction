"""项目公共配置"""

import os

# ---------- 标的 ----------
STOCK_CODE = "sz.300454"      # baostock 格式代码
STOCK_SYMBOL = "300454"       # akshare / 无前缀格式代码
STOCK_NAME = "深信服"
INDEX_CODE = "sz.399006"      # 创业板指(深信服所属板块基准)

# ---------- 数据时间范围 ----------
HISTORY_START_DATE = "2010-01-01"   # 足够早,baostock会自动从实际上市日开始返回
TRAIN_TEST_SPLIT_DATE = "2025-01-01"  # 早于此日期用于训练,此日期及以后用于walk-forward回测

# ---------- 路径 ----------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

HISTORY_CSV = os.path.join(DATA_DIR, "history.csv")
FUNDAMENTALS_CSV = os.path.join(DATA_DIR, "fundamentals.csv")
INDUSTRY_CSV = os.path.join(DATA_DIR, "industry.csv")
INDEX_CSV = os.path.join(DATA_DIR, "index.csv")
SENTIMENT_CSV = os.path.join(DATA_DIR, "sentiment.csv")

BACKTEST_RESULTS_CSV = os.path.join(OUTPUTS_DIR, "backtest_results.csv")
PREDICTIONS_CSV = os.path.join(OUTPUTS_DIR, "predictions.csv")
ACCURACY_REPORT_MD = os.path.join(REPORTS_DIR, "accuracy_report.md")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------- 随机种子 ----------
RANDOM_SEED = 42

# ---------- 网络请求 ----------
# akshare(东方财富等)接口在部分网络环境下若不带浏览器UA会被连接重置,统一走这个UA
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def patch_requests_user_agent():
    """给requests.get打补丁,强制带上浏览器UA(akshare内部请求不带UA,部分网络会拒绝)"""
    import requests

    if getattr(requests.get, "_ua_patched", False):
        return
    _orig_get = requests.get

    def _get_with_ua(*args, **kwargs):
        headers = kwargs.get("headers") or {}
        headers.setdefault("User-Agent", BROWSER_USER_AGENT)
        kwargs["headers"] = headers
        return _orig_get(*args, **kwargs)

    _get_with_ua._ua_patched = True
    requests.get = _get_with_ua


def fix_console_encoding():
    """Windows终端默认GBK,打印中文会乱码,这里强制标准输出用UTF-8"""
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")


def patch_pandas_for_akshare():
    """pandas 3.0 默认的 infer_string 会导致部分 akshare 老接口的正则表达式报 ArrowInvalid,这里关掉"""
    import pandas as pd

    try:
        pd.set_option("future.infer_string", False)
    except Exception:
        pass
