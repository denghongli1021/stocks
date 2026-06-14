"""🔀 多檔比較頁:把多檔股票走勢「歸一化」後疊在一起比較相對強弱。"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.webutils import POPULAR, label_of, load_data

st.title("🔀 多檔股票比較")
st.caption("把各股票收盤價歸一化到起點 = 100,方便直接比較「同期間漲幅」,不受股價高低影響。")

all_syms = [s for s, _ in POPULAR]
col = st.columns([3, 2])
selected = col[0].multiselect(
    "選擇要比較的股票", options=all_syms, default=["2330.TW", "NVDA", "AAPL"],
    format_func=label_of,
)
period = col[1].selectbox("期間", ["6mo", "1y", "2y", "5y"], index=1)

extra = st.text_input("另外加入股票(代號以逗號分隔,例如 2603.TW, MSFT)", value="")
symbols = list(selected) + [x.strip() for x in extra.split(",") if x.strip()]
symbols = list(dict.fromkeys(symbols))  # 去重、保序

if not symbols:
    st.info("請至少選擇一檔股票。")
    st.stop()

# 抓資料
data, failed = {}, []
with st.spinner("抓取資料中..."):
    for sym in symbols:
        try:
            data[sym] = load_data(sym, period)
        except Exception:  # noqa: BLE001
            failed.append(sym)
if failed:
    st.warning("以下代號抓取失敗,已略過:" + "、".join(failed))
if not data:
    st.stop()

# ---- 歸一化疊圖 ----
fig = go.Figure()
rows = []
for sym, df in data.items():
    close = df["Close"].dropna()
    norm = close / close.iloc[0] * 100
    fig.add_trace(go.Scatter(x=norm.index, y=norm, name=label_of(sym)))
    ret = close.pct_change()
    rows.append({
        "股票": label_of(sym),
        "期間報酬": f"{(close.iloc[-1] / close.iloc[0] - 1) * 100:+.1f}%",
        "年化波動": f"{ret.std() * np.sqrt(252) * 100:.1f}%",
        "最新價": round(float(close.iloc[-1]), 2),
    })
fig.add_hline(y=100, line_dash="dot", line_color="#888")
fig.update_layout(height=520, yaxis_title="歸一化價格(起點=100)",
                  legend=dict(orientation="h", y=1.02), margin=dict(t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---- 數據表 ----
st.subheader("期間表現")
st.dataframe(pd.DataFrame(rows).set_index("股票"), use_container_width=True)
st.caption("年化波動 = 日報酬標準差 × √252,數字越大代表股價起伏越劇烈。")
