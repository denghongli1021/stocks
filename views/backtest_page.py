"""🧪 回測頁:用模型的隔日報酬預測當訊號,模擬交易並與買入持有比較。"""
import plotly.graph_objects as go
import streamlit as st

from src.backtest import run_backtest
from src.webutils import POPULAR, get_model, label_of, load_data

st.title("🧪 策略回測")
st.caption(
    "策略:模型預測「隔日報酬 > 門檻」就做多(持有一天),否則空手;每次換倉扣交易成本。"
    "全程只用樣本外的測試段資料,並與「買入持有」對照。"
)

trained = [s for s, _ in POPULAR if get_model(s) is not None]
if not trained:
    st.warning("目前沒有已訓練的模型。請先到「個股預測」頁訓練至少一檔。")
    st.stop()

col = st.columns([2, 1, 1, 1])
ticker = col[0].selectbox("選擇股票(需已訓練)", trained, format_func=label_of)
threshold = col[1].number_input("做多門檻(%)", value=0.0, step=0.05, format="%.2f") / 100
cost = col[2].number_input("交易成本(%)", value=0.1, step=0.05, format="%.2f") / 100
allow_short = col[3].checkbox("允許做空", value=False)

if not st.button("▶️ 執行回測", type="primary"):
    st.info("設定好參數後按「執行回測」。")
    st.stop()

model = get_model(ticker)
with st.spinner("載入資料並逐日預測中..."):
    df = load_data(ticker, "8y")
    pred_df = model.predict_series(df, only_test=True)
    daily, m = run_backtest(pred_df, threshold=threshold, cost=cost, allow_short=allow_short)

# ---- 權益曲線 ----
st.subheader("權益曲線(初始資金 = 1)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily.index, y=daily["equity"], name="策略",
                         line=dict(color="#d62728")))
fig.add_trace(go.Scatter(x=daily.index, y=daily["bh_equity"], name="買入持有",
                         line=dict(color="#1f77b4")))
fig.update_layout(height=440, legend=dict(orientation="h", y=1.02), margin=dict(t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---- 指標對照 ----
st.subheader("績效對照")
c1, c2, c3, c4 = st.columns(4)
c1.metric("策略總報酬", f"{m['strat_total'] * 100:+.1f}%",
          f"買入持有 {m['bh_total'] * 100:+.1f}%")
c2.metric("策略年化", f"{m['strat_ann'] * 100:+.1f}%",
          f"買入持有 {m['bh_ann'] * 100:+.1f}%")
c3.metric("策略 Sharpe", f"{m['strat_sharpe']:.2f}",
          f"買入持有 {m['bh_sharpe']:.2f}")
c4.metric("策略最大回撤", f"{m['strat_mdd'] * 100:.1f}%",
          f"買入持有 {m['bh_mdd'] * 100:.1f}%", delta_color="inverse")

c5, c6, c7 = st.columns(3)
c5.metric("勝率(進場日)", f"{m['win_rate'] * 100:.1f}%")
c6.metric("交易次數", f"{m['n_trades']}")
c7.metric("持倉時間比例", f"{m['exposure'] * 100:.0f}%")
st.caption(f"回測樣本:{m['days']} 個交易日（{daily.index[0].date()} ~ {daily.index[-1].date()}）")

# ---- 誠實提醒 ----
if m["strat_total"] < m["bh_total"]:
    st.info(
        "📌 策略多半**贏不過買入持有**——這正是預期結果。模型方向準確率約 50%,"
        "扣掉交易成本後很難勝過大盤。這個回測的價值在於**誠實驗證**:它示範了"
        "「價格預測看起來很準」不等於「能賺錢」,提醒不要被漂亮的預測曲線誤導。",
        icon="🧠",
    )
else:
    st.info(
        "📌 此區間策略略勝買入持有,但這可能只是**特定期間的運氣**。換股票或換時間"
        "常常就反轉,且未計入滑價、稅費。切勿據此實際交易。",
        icon="🧠",
    )
