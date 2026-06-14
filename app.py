"""股價走勢預測 — 多頁網站入口。

啟動:
    streamlit run app.py

三個頁面:
    📊 個股預測   走勢 + LSTM 預測
    🔀 多檔比較   多檔股票走勢疊圖
    🧪 回測       用預測訊號模擬交易,與買入持有比較
"""
import streamlit as st

st.set_page_config(page_title="股價走勢預測", page_icon="📈", layout="wide")

pages = [
    st.Page("views/predict.py", title="個股預測", icon="📊", default=True),
    st.Page("views/compare.py", title="多檔比較", icon="🔀"),
    st.Page("views/backtest_page.py", title="回測", icon="🧪"),
]
st.navigation(pages).run()
