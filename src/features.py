"""技術指標 / 特徵工程。

刻意使用「平穩型」特徵:報酬率、乖離率、振盪指標、波動率、量能比值。
對趨勢股(價格不斷創新高)也能良好縮放,且涵蓋動量 / 量能 / 波動率多面向。
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
    return close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """平均真實波幅 ATR:衡量波動度。真實波幅取 高低差 / 與昨收差 的最大值。"""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在 OHLCV 上加上「平穩型」技術指標,丟掉開頭因計算而產生的 NaN 列。"""
    out = df.copy()
    close, high, low = out["Close"], out["High"], out["Low"]
    volume = out["Volume"].replace(0, np.nan).ffill().fillna(1.0)

    # --- 動量 ---
    out["Return"] = np.log(close / close.shift(1))           # 日 log 報酬
    out["Return5"] = np.log(close / close.shift(5))           # 5 日累積報酬
    out["ROC10"] = close / close.shift(10) - 1               # 10 日變動率

    # --- 均線乖離 ---
    out["MA5_ratio"] = close / close.rolling(5).mean() - 1
    out["MA10_ratio"] = close / close.rolling(10).mean() - 1
    out["MA20_ratio"] = close / close.rolling(20).mean() - 1
    out["MA60_ratio"] = close / close.rolling(60).mean() - 1

    # --- 振盪指標 ---
    out["RSI14n"] = rsi(close, 14) / 100.0                   # RSI 正規化 0~1
    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    out["StochK"] = (close - low14) / (high14 - low14).replace(0, 1e-10)  # 隨機指標 %K
    out["MACD_norm"] = macd(close) / close                   # MACD 以價格正規化
    std20 = close.rolling(20).std()
    out["BB_pos"] = (close - close.rolling(20).mean()) / (2 * std20.replace(0, 1e-10))  # 布林帶位置

    # --- 波動率 ---
    out["Vol20"] = out["Return"].rolling(20).std()           # 20 日報酬波動
    out["ATR14n"] = atr(out, 14) / close                     # ATR 以價格正規化
    out["HL_range"] = (high - low) / close                   # 當日高低振幅

    # --- 量能(交易量) ---
    out["VolChange"] = np.log(volume / volume.shift(1))      # 量變化(log)
    out["VolRatio"] = volume / volume.rolling(20).mean() - 1  # 相對 20 日均量

    return out.replace([np.inf, -np.inf], np.nan).dropna()
