"""抓取股價資料,並做本地 CSV 快取。"""
from __future__ import annotations

import os
import time
import pandas as pd
import yfinance as yf

import config


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 單一股票會回傳 MultiIndex 欄位 (Price, Ticker),這裡攤平成單層。"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_data(
    ticker: str = config.TICKER,
    period: str = config.PERIOD,
    interval: str = config.INTERVAL,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """抓取 OHLCV 日線資料。

    回傳的 DataFrame 以日期為 index,欄位含 Open/High/Low/Close/Volume。
    第一次抓會存到 data/{ticker}_{interval}.csv,之後預設讀快取(force_refresh=True 可強制重抓)。
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tag = f"{ticker.replace('.', '_')}_{interval}_{period}"
    cache_path = os.path.join(config.DATA_DIR, f"{tag}.csv")

    if use_cache and not force_refresh and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if len(df) > 0:
            return df

    # yfinance 在部分環境會偶發 TLS 連線錯誤,這裡重試幾次提高穩定度
    df = pd.DataFrame()
    last_err = None
    for attempt in range(4):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,   # 自動還原權息,長期資料更合理
                progress=False,
            )
            df = _flatten_columns(df).dropna()
            if not df.empty:
                break
        except Exception as e:  # noqa: BLE001 — 網路層各種例外都重試
            last_err = e
        time.sleep(1.5 * (attempt + 1))

    if df.empty:
        raise ValueError(
            f"抓不到 {ticker} 的資料(重試多次仍失敗)。請確認代號正確"
            f"(台股要加 .TW,例如 2330.TW)且有網路。最後錯誤:{last_err}"
        )

    # 只保留需要的欄位、確保型別
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].astype("float64")
    df.index.name = "Date"
    df.to_csv(cache_path)
    return df


if __name__ == "__main__":
    # 直接執行可快速測試抓資料
    data = fetch_data(force_refresh=True)
    print(f"{config.TICKER}: 共 {len(data)} 筆,{data.index.min().date()} ~ {data.index.max().date()}")
    print(data.tail())
