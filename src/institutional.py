"""三大法人買賣超(外資 / 投信 / 自營商)資料 —— 僅台股適用,資料來源 FinMind。

把每日各法人的買賣超(股數)正規化成「相對近 20 日均量」的比值,作為平穩特徵,
反映大戶資金流向。非台股(美股等)會回傳 None,模型自動只用技術特徵。
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

import config

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanStockInstitutionalInvestorsBuySell"

# 法人類別歸併:外資、投信、自營商
_FOREIGN = {"Foreign_Investor", "Foreign_Dealer_Self"}
_TRUST = {"Investment_Trust"}
_DEALER = {"Dealer_self", "Dealer_Hedging"}

# 最終輸出的法人特徵欄位
INST_FEATURES = ["Foreign_ratio", "Foreign_ratio5", "Trust_ratio", "Dealer_ratio"]


def is_tw_ticker(ticker: str) -> bool:
    return ticker.upper().endswith((".TW", ".TWO"))


def _stock_id(ticker: str) -> str:
    return ticker.split(".")[0]


def fetch_institutional(ticker: str, start: str = "2015-01-01",
                        force_refresh: bool = False) -> pd.DataFrame | None:
    """抓三大法人買賣超(股數),回傳 index=date、欄位 Foreign_net/Trust_net/Dealer_net。

    非台股回傳 None;抓取失敗(網路 / 限流)也回傳 None,讓模型優雅退化成只用技術特徵。
    """
    if not is_tw_ticker(ticker):
        return None

    os.makedirs(config.DATA_DIR, exist_ok=True)
    cache = os.path.join(config.DATA_DIR, f"{_stock_id(ticker)}_institutional.csv")
    if not force_refresh and os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if len(df):
            return df

    params = {"dataset": DATASET, "data_id": _stock_id(ticker),
              "start_date": start, "end_date": "2030-12-31"}
    raw = pd.DataFrame()
    for attempt in range(3):
        try:
            r = requests.get(FINMIND_URL, params=params, timeout=60)
            j = r.json()
            if j.get("msg") == "success" and j.get("data"):
                raw = pd.DataFrame(j["data"])
                break
        except Exception:  # noqa: BLE001 — 網路 / JSON 各種例外都重試
            pass
        time.sleep(1.5 * (attempt + 1))

    if raw.empty:
        return None

    raw["date"] = pd.to_datetime(raw["date"])
    raw["net"] = raw["buy"].astype("float64") - raw["sell"].astype("float64")

    def _sum(cats):
        return (raw[raw["name"].isin(cats)].groupby("date")["net"].sum())

    out = pd.DataFrame({
        "Foreign_net": _sum(_FOREIGN),
        "Trust_net": _sum(_TRUST),
        "Dealer_net": _sum(_DEALER),
    }).fillna(0.0).sort_index()
    out.index.name = "Date"
    out.to_csv(cache)
    return out


def institutional_features(price_df: pd.DataFrame, ticker: str,
                           force_refresh: bool = False) -> pd.DataFrame | None:
    """把法人買賣超正規化成平穩特徵(相對近 20 日均量),對齊到 price_df 的日期。

    回傳含 INST_FEATURES 欄位的 DataFrame(index 與 price_df 對齊);抓不到則回傳 None。
    """
    raw = fetch_institutional(ticker, force_refresh=force_refresh)
    if raw is None or raw.empty:
        return None

    raw = raw.reindex(price_df.index).fillna(0.0)              # 對齊交易日,缺日當 0
    denom = price_df["Volume"].rolling(20).mean().replace(0, np.nan)  # 近 20 日均量

    out = pd.DataFrame(index=price_df.index)
    out["Foreign_ratio"] = raw["Foreign_net"] / denom
    out["Foreign_ratio5"] = raw["Foreign_net"].rolling(5).sum() / (denom * 5)  # 5 日累積外資
    out["Trust_ratio"] = raw["Trust_net"] / denom
    out["Dealer_ratio"] = raw["Dealer_net"] / denom
    return out.replace([np.inf, -np.inf], np.nan)
