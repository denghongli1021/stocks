"""命令列訓練腳本:抓資料 -> 訓練 LSTM -> 存模型 -> 印出測試指標。

用法:
    python train.py                      # 用 config 預設(台積電 2330.TW)
    python train.py --ticker 2330.TW --epochs 60 --refresh
"""
from __future__ import annotations

import argparse
import sys

# Windows 主控台預設可能是 cp950(Big5),Keras 進度條與中文輸出會編碼失敗;
# 強制 stdout 改用 utf-8(編不出的字以 replace 取代,確保永不崩潰)。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import config
from src.data_loader import fetch_data
from src.model import StockModel


def main():
    parser = argparse.ArgumentParser(description="訓練股價預測 LSTM 模型")
    parser.add_argument("--ticker", default=config.TICKER, help="股票代號,台股需加 .TW")
    parser.add_argument("--period", default=config.PERIOD, help="抓取期間,例如 5y / 8y / max")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--refresh", action="store_true", help="強制重新抓資料(不讀快取)")
    args = parser.parse_args()

    print(f"[1/3] 抓取 {args.ticker} 資料(period={args.period})...")
    df = fetch_data(args.ticker, period=args.period, force_refresh=args.refresh)
    print(f"      共 {len(df)} 筆,{df.index.min().date()} ~ {df.index.max().date()}")

    print(f"[2/3] 訓練 LSTM(epochs={args.epochs})...")
    sm = StockModel(ticker=args.ticker)
    # verbose=2:每個 epoch 印一行,不用會觸發編碼錯誤的動態進度條
    sm.fit(df, epochs=args.epochs, verbose=2)

    print(f"[3/3] 儲存模型至 {config.MODEL_DIR}/ ...")
    sm.save()

    print("\n=== 測試集表現 ===")
    for k, v in sm.metrics.items():
        unit = "%" if k in ("MAPE", "DirectionAcc") else ""
        print(f"  {k:14s}: {v:.4f}{unit}")

    fc = sm.forecast(df, days=5)
    print("\n=== 未來 5 個交易日預測 ===")
    print(fc.round(2).to_string())


if __name__ == "__main__":
    main()
