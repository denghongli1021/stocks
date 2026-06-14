"""技術指標 / 特徵工程。

刻意使用「平穩型」特徵:報酬率、價格相對均線的比值、波動率、正規化後的 RSI/MACD。
這樣對趨勢股(價格不斷創新高)也能良好縮放,且未來遞迴預測時可由收盤價自我重算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """相對強弱指標 RSI(0~100)。> 70 偏超買、< 30 偏超賣。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD 線 = 12 日 EMA - 26 日 EMA。"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在 OHLCV 上加上「平穩型」技術指標欄位,丟掉開頭因計算而產生的 NaN 列。"""
    out = df.copy()
    close = out["Close"]

    out["Return"] = np.log(close / close.shift(1))          # 日 log 報酬率
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    out["MA5_ratio"] = close / ma5 - 1                       # 價格相對 5 日均線
    out["MA10_ratio"] = close / ma10 - 1
    out["MA20_ratio"] = close / ma20 - 1
    out["RSI14n"] = rsi(close, 14) / 100.0                   # RSI 正規化到 0~1
    out["MACD_norm"] = macd(close) / close                  # MACD 以價格正規化
    out["Vol20"] = out["Return"].rolling(20).std()          # 20 日波動率

    return out.dropna()
