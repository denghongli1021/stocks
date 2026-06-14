"""📊 個股預測頁:輸入代號 → 顯示走勢 → 顯示 LSTM 預測。"""
import plotly.graph_objects as go
import streamlit as st

import config
from src.model import StockModel
from src.webutils import POPULAR, StProgress, candlestick, get_model, load_data

# ---------- 側邊欄(進階設定) ----------
st.sidebar.header("⚙️ 進階設定")
period = st.sidebar.selectbox("歷史資料長度", ["1y", "2y", "5y", "8y", "max"], index=3)
epochs = st.sidebar.slider("訓練 Epochs", 10, 120, 40, step=10)
forecast_days = st.sidebar.slider("預測未來幾個交易日", 1, 30, 5)
refresh = st.sidebar.checkbox("強制重新抓資料", value=False)

# ---------- 標題 + 熱門按鈕 + 搜尋列 ----------
st.title("📈 股價走勢預測")

st.caption("熱門股票(已預先訓練,點了秒出結果):")
pcols = st.columns(len(POPULAR))
for col, (sym, label) in zip(pcols, POPULAR):
    if col.button(label, use_container_width=True, key=f"pop_{sym}"):
        st.session_state.ticker = sym

search = st.columns([4, 1])
ticker = search[0].text_input(
    "輸入股票代號", value=st.session_state.get("ticker", config.TICKER),
    placeholder="例如 2330.TW(台積電)、NVDA(輝達)", label_visibility="collapsed",
).strip()
search[1].button("🔍 查詢", use_container_width=True, type="primary")

st.warning("⚠️ 本工具僅供學習研究,**不構成投資建議**。", icon="⚠️")

if not ticker:
    st.info("請在上方輸入股票代號後按「查詢」。")
    st.stop()

# ---------- 抓資料(輸入代號就自動顯示走勢) ----------
try:
    with st.spinner(f"抓取 {ticker} 資料中..."):
        df = load_data(ticker, period, refresh)
    st.session_state.ticker = ticker
except Exception as e:  # noqa: BLE001
    st.error(f"抓不到「{ticker}」的資料,請確認代號正確(台股要加 .TW)。\n\n{e}")
    st.stop()

# ============================================================
#  區塊一:走勢
# ============================================================
st.header(f"📊 {ticker} 走勢")
last, prev = df.iloc[-1], df.iloc[-2]
change = last["Close"] - prev["Close"]
pct = change / prev["Close"] * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("最新收盤", f"{last['Close']:.2f}", f"{change:+.2f} ({pct:+.2f}%)")
c2.metric("資料筆數", f"{len(df):,}")
c3.metric("區間最高", f"{df['Close'].max():.2f}")
c4.metric("區間最低", f"{df['Close'].min():.2f}")
st.plotly_chart(candlestick(df), use_container_width=True)

# ============================================================
#  區塊二:預測
# ============================================================
st.header("🔮 預測")
model = get_model(ticker)

cta = st.columns([1, 1, 3])
do_train = cta[0].button(
    "🚀 訓練模型" if model is None else "🔁 重新訓練", type="primary", use_container_width=True)
do_predict = cta[1].button("📈 重新預測未來", use_container_width=True) if model else False

if model is None and not do_train:
    st.info(f"「{ticker}」尚無訓練好的模型,按上方「🚀 訓練模型」即可開始(約數十秒到數分鐘)。")
    st.stop()

if do_train:
    sm = StockModel(ticker=ticker)
    with st.status("訓練 LSTM 模型中...", expanded=True) as status:
        sm.fit(df, epochs=epochs, extra_callbacks=[StProgress(epochs)], verbose=0)
        sm.save()
        status.update(label="✅ 訓練完成,模型已儲存到 models/", state="complete")
    st.session_state.setdefault("model_cache", {})[ticker] = sm
    model = sm

if model is None:
    st.stop()

# ---- 評估指標 ----
if model.metrics:
    m = model.metrics
    st.subheader("測試集表現")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("RMSE", f"{m['RMSE']:.2f}", help="價格均方根誤差,越小越好")
    k2.metric("MAE", f"{m['MAE']:.2f}", help="平均絕對誤差(價格)")
    k3.metric("MAPE", f"{m['MAPE']:.2f}%", help="平均絕對百分比誤差")
    k4.metric("方向準確率", f"{m['DirectionAcc']:.1f}%",
              help="預測漲跌方向正確的比例;接近 50% 屬正常(短期方向極難預測)")

# ---- 測試集:真實 vs 預測 ----
if model.test_dates is not None:
    st.subheader("測試集:真實 vs 預測")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=model.test_dates, y=model.test_true,
                              name="真實收盤", line=dict(color="#1f77b4")))
    fig2.add_trace(go.Scatter(x=model.test_dates, y=model.test_pred,
                              name="模型預測", line=dict(color="#ff7f0e", dash="dot")))
    fig2.update_layout(height=380, legend=dict(orientation="h", y=1.02), margin=dict(t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ---- 未來預測 ----
st.subheader(f"未來 {forecast_days} 個交易日預測")
with st.spinner("遞迴推估未來走勢中..."):
    fc = model.forecast(df, days=forecast_days)

recent = df["Close"].iloc[-60:]
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=recent.index, y=recent.values,
                          name="近 60 日實際", line=dict(color="#1f77b4")))
fig3.add_trace(go.Scatter(x=[recent.index[-1], fc.index[0]],
                          y=[recent.values[-1], fc["PredictedClose"].iloc[0]],
                          line=dict(color="#d62728", dash="dot"), showlegend=False))
fig3.add_trace(go.Scatter(x=fc.index, y=fc["PredictedClose"], name="未來預測",
                          mode="lines+markers", line=dict(color="#d62728", dash="dot")))
fig3.update_layout(height=420, legend=dict(orientation="h", y=1.02), margin=dict(t=30, b=10))
st.plotly_chart(fig3, use_container_width=True)

pred_change = fc["PredictedClose"].iloc[-1] - last["Close"]
pred_pct = pred_change / last["Close"] * 100
st.metric(f"{forecast_days} 個交易日後預測收盤",
          f"{fc['PredictedClose'].iloc[-1]:.2f}", f"{pred_change:+.2f} ({pred_pct:+.2f}%)")

show = fc.copy()
show["PredictedClose"] = show["PredictedClose"].round(2)
show.index = show.index.date
st.dataframe(show, use_container_width=True)
st.caption("註:未來為「遞迴」推估(用預測值當作下一天輸入),越往後誤差累積越大,僅供參考。")
