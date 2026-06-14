# 📈 股價走勢預測網站

用 **LSTM 深度學習** 預測股價(預設台積電 2330.TW),做成 **Streamlit 多頁互動網站**。
涵蓋完整流程:**抓資料 → 特徵工程 → 訓練 → 評估 → 預測 → 多檔比較 → 回測**。

> ⚠️ **免責聲明**:本專案僅供學習與研究,**不構成任何投資建議**。股市有風險,過去績效不代表未來。

---

## 三個頁面

| 頁面 | 功能 |
|------|------|
| 📊 **個股預測** | 輸入代號 → 顯示走勢 K 線 → LSTM 預測隔日 / 未來多日;附 6 檔熱門股一鍵按鈕(已預訓練) |
| 🔀 **多檔比較** | 多檔股票歸一化疊圖,比較同期間相對強弱與波動 |
| 🧪 **回測** | 用預測訊號模擬交易,與「買入持有」比較總報酬、Sharpe、最大回撤 |

---

## 快速開始

```bash
pip install -r requirements.txt    # 安裝套件
streamlit run app.py               # 啟動網站(瀏覽器開 http://localhost:8501)
```

(可選)用命令列先訓練模型:

```bash
python train.py --ticker 2330.TW --period 8y --epochs 60
```

---

## 專案結構

```
stocks/
├── app.py                    # 入口:多頁導覽(st.navigation)
├── views/
│   ├── predict.py            # 📊 個股預測頁
│   ├── compare.py            # 🔀 多檔比較頁
│   └── backtest_page.py      # 🧪 回測頁
├── src/
│   ├── data_loader.py        # yfinance 抓 OHLCV + 快取 + 自動重試
│   ├── features.py           # 平穩型技術指標
│   ├── model.py              # LSTM 訓練 / 評估 / 預測 / 逐日預測 / 存讀檔
│   ├── backtest.py           # 回測引擎(純邏輯)
│   └── webutils.py           # Streamlit 共用工具(資料載入、圖表、熱門清單)
├── train.py                  # 命令列訓練腳本
├── config.py                 # 所有可調參數
├── models/                   # 預先訓練好的模型(已隨專案附上,可直接用)
└── requirements.txt
```

---

## 核心原理(重點摘要)

### 為什麼預測「報酬率」而非「價格」?
台積電等趨勢股價格不斷創新高。若直接縮放絕對價格並預測價格,測試期超出訓練期看過的範圍,
縮放器飽和、模型無法外推 → **系統性低估**(實測 MAPE ~19%)。
改用**平穩特徵**(報酬率、價格/均線比值、波動率…)並**預測隔日 log 報酬**再還原價格後,
測試 **MAPE 降到 ~1.5%**。

### 進階特徵(技術 16 個)
涵蓋多面向訊號,全部平穩化:
- **動量**:日報酬、5 日報酬、10 日 ROC
- **均線乖離**:價格 / MA5、MA10、MA20、MA60
- **振盪指標**:RSI、隨機指標 %K、MACD、布林帶位置
- **波動率**:20 日報酬波動、ATR、當日高低振幅
- **量能(交易量)**:量變化、相對 20 日均量

### 三大法人特徵(台股 +4 個,資料來源 [FinMind](https://finmindtrade.com/))
**僅台股**會額外加入(美股無此資料,自動只用技術特徵):
- **外資**:當日買賣超 / 近 20 日均量、5 日累積外資買賣超
- **投信**:買賣超 / 均量
- **自營商**:買賣超 / 均量

> 為什麼加這個?三大法人(尤其外資)的買賣超反映**大戶實際資金流向**,是價量技術指標
> 之外的「新資訊」,比再疊技術指標更可能帶來真實 edge。每檔股票模型會記住自己用了
> 哪些特徵(台股 20 個、美股 16 個)。可在 `config.py` 用 `USE_INSTITUTIONAL` 開關。

> 實測(台積電,同切分同種子):加入量能等進階特徵後,**方向準確率從 ~47% → ~52%**,
> MAPE 維持 ~1.5%。方向有改善,但仍接近 50%——短期漲跌本質極難預測。

### 模型
兩層 LSTM + Dropout,輸入過去 60 天特徵序列,Huber loss。時間序列**依時間切三段
(訓練 70% / 驗證 15% / 測試 15%)**:早停看驗證集、評估看測試集,縮放器**只用訓練段 fit**
(避免未來資訊洩漏)。未來多日採**遞迴**預測。

### 誠實提醒
- 方向準確率約 **50%**(接近擲硬幣)——短期漲跌本質極難預測。
- 回測常顯示策略**贏不過買入持有**(尤其扣成本後)。這正是重點:**「預測曲線好看」≠「能賺錢」**。

---

## 🚀 部署上線(Streamlit Community Cloud,免費)

1. **建 GitHub repo 並推上去**(在專案資料夾):
   ```bash
   git init
   git add -A
   git commit -m "台積電股價預測網站"
   git branch -M main
   git remote add origin https://github.com/<你的帳號>/<repo 名>.git
   git push -u origin main
   ```
   > 已附 `.gitignore`:會上傳 `models/`(預訓練模型),但不上傳 `data/` 快取。

2. **到 [share.streamlit.io](https://share.streamlit.io)** 用 GitHub 登入 → **New app**:
   - Repository:選你剛推的 repo
   - Branch:你的分支(`main` 或 `master`)
   - Main file path:`app.py`
   - ⚠️ **務必展開 Advanced settings → Python version 選 `3.12`**
     (TensorFlow **只支援到 3.12**;預設常是最新的 3.13/3.14,沒有 TF 安裝檔會直接 build 失敗)

3. 按 **Deploy**,等套件安裝(TensorFlow 較大,首次約數分鐘),完成後會得到一個
   `https://<你的app>.streamlit.app` 的公開網址,手機也能開。

> 💡 免費方案資源有限(~1GB RAM)。本專案模型小、6 檔已預訓練,通常可順利運行;
> 若資源吃緊,可在 `views/predict.py` 把預設 `period` 改短一點。

### 🛠 部署疑難排解

**`No matching distribution found for tensorflow-cpu`**(或 `No solution found when resolving dependencies`)
→ 代表雲端用了 **Python 3.13/3.14**,TensorFlow 還沒有對應安裝檔。**改用 Python 3.12**:

- 既有 App:右下角 **Manage app → ⋮ → Settings → Python version 改 3.12 → Reboot**;
  若設定頁沒有版本選項,就 **Delete app 後重新 Deploy**,在 Advanced settings 選 3.12。
- 重新部署前確認本 repo 的 `requirements.txt` 已是收緊版本(本專案已設定好)。

---

## 延伸方向
- 更多資料源:三大法人買賣超、新聞情緒、總經指標。
- 換模型:GRU、Transformer、TFT;或加分位數預測給出「區間」。
- 把回測做得更嚴謹:滑價、停損停利、部位控管、walk-forward 重訓。
