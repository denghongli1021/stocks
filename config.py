"""全域設定檔。所有可調參數集中在這裡,方便修改。"""

# ---- 股票 ----
TICKER = "2330.TW"        # 台積電(台股代號 2330,.TW 代表台灣證交所)
PERIOD = "8y"             # 抓多久的歷史資料(yfinance 格式:1y, 5y, 8y, max...)
INTERVAL = "1d"           # K 線週期:1d(日)、1wk(週)、1h(小時)

# ---- 特徵 ----
# 全部用「平穩型」特徵(報酬率、比值、波動率、量能…),不直接餵絕對價格。
# 趨勢股(如台積電)的價格會不斷創新高,直接縮放絕對價格會讓模型無法外推;
# 改用平穩特徵 + 預測「報酬率」可大幅改善。
# 進階特徵涵蓋:多週期動量、量能(交易量)、波動率(ATR)、布林帶位置、隨機指標。
FEATURES = [
    "Return", "Return5", "ROC10",                          # 動量
    "MA5_ratio", "MA10_ratio", "MA20_ratio", "MA60_ratio",  # 均線乖離
    "RSI14n", "StochK", "MACD_norm", "BB_pos",              # 振盪指標
    "Vol20", "ATR14n", "HL_range",                         # 波動率
    "VolChange", "VolRatio",                               # 量能(交易量)
]
# 預測目標:隔日 log 報酬率(模型內部處理),再還原成價格。

# ---- 模型 / 訓練 ----
LOOKBACK = 60             # 用過去 60 天的序列,預測第 61 天
# 時間序列「依時間切」三段:訓練 / 驗證 / 測試(不可隨機打亂)。
# 早停看「驗證集」、最終評估看「測試集」,避免用測試集做模型挑選的偏誤。
TRAIN_RATIO = 0.70        # 前 70% 訓練
VAL_RATIO = 0.15          # 中間 15% 驗證(剩下 15% 為測試)
EPOCHS = 80
BATCH_SIZE = 32
LSTM_UNITS = 64
DROPOUT = 0.2
LEARNING_RATE = 1e-3
PATIENCE = 15             # EarlyStopping:驗證集連續沒進步就提早停

# ---- 路徑 ----
MODEL_DIR = "models"
DATA_DIR = "data"
