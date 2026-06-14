"""Streamlit 各頁面共用的工具:資料載入、模型取得、圖表、熱門清單。"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import tensorflow as tf

import config
from src.data_loader import fetch_data
from src.model import StockModel

# 熱門股票(已預先訓練,點了秒出結果)
POPULAR = [
    ("2330.TW", "台積電"), ("2454.TW", "聯發科"), ("2317.TW", "鴻海"),
    ("2308.TW", "台達電"), ("NVDA", "輝達"), ("AAPL", "蘋果"),
]
LABEL = {sym: f"{label}（{sym}）" for sym, label in POPULAR}


def label_of(sym: str) -> str:
    return LABEL.get(sym, sym)


class StProgress(tf.keras.callbacks.Callback):
    """把 Keras 每個 epoch 的進度顯示成 Streamlit 進度條。"""

    def __init__(self, total_epochs: int):
        super().__init__()
        self.total = total_epochs
        self.bar = st.progress(0.0)
        self.text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        frac = (epoch + 1) / self.total
        self.bar.progress(min(frac, 1.0))
        self.text.caption(
            f"Epoch {epoch + 1}/{self.total} — "
            f"loss={logs.get('loss', 0):.4f}, val_loss={logs.get('val_loss', 0):.4f}"
        )


@st.cache_data(show_spinner=False)
def load_data(ticker: str, period: str, refresh: bool = False) -> pd.DataFrame:
    return fetch_data(ticker, period=period, force_refresh=refresh)


def get_model(ticker: str) -> StockModel | None:
    """從 session 取模型;沒有就試著從磁碟載入已訓練模型。"""
    cache = st.session_state.setdefault("model_cache", {})
    if ticker in cache:
        return cache[ticker]
    if StockModel.exists(ticker):
        try:
            cache[ticker] = StockModel.load(ticker)
            return cache[ticker]
        except Exception:  # noqa: BLE001
            return None
    return None


def candlestick(df: pd.DataFrame, with_rangeselector: bool = True) -> go.Figure:
    """K 線 + MA20/MA60 + 成交量的組合圖。"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(),
                             line=dict(width=1), name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(60).mean(),
                             line=dict(width=1), name="MA60"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="成交量",
                         marker_color="#9aa0a6"), row=2, col=1)
    fig.update_layout(height=540, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02), margin=dict(t=30, b=10))
    if with_rangeselector:
        fig.update_xaxes(rangeselector=dict(buttons=[
            dict(count=1, label="1月", step="month", stepmode="backward"),
            dict(count=3, label="3月", step="month", stepmode="backward"),
            dict(count=6, label="6月", step="month", stepmode="backward"),
            dict(count=1, label="1年", step="year", stepmode="backward"),
            dict(step="all", label="全部"),
        ]), row=1, col=1)
    return fig
