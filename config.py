"""全域設定檔。所有可調參數集中在這裡,方便修改。"""

# ---- 股票 ----
TICKER = "2330.TW"        # 台積電(台股代號 2330,.TW 代表台灣證交所)
PERIOD = "8y"             # 抓多久的歷史資料(yfinance 格式:1y, 5y, 8y, max...)
INTERVAL = "1d"           # K 線週期:1d(日)、1wk(週)、1h(小時)

# ---- 特徵 ----
# 全部用「平穩型」特徵(報酬率、價格相對均線的比值、波動率…),不直接餵絕對價格。
# 趨勢股(如台積電)的價格會不斷創新高,直接縮放絕對價格會讓模型無法外推;
# 改用平穩特徵 + 預測「報酬率」可大幅改善。
FEATURES = ["Return", "MA5_ratio", "MA10_ratio", "MA20_ratio", "RSI14n", "MACD_norm", "Vol20"]
# 預測目標:隔日 log 報酬率(模型內部處理),再還原成價格。

# ---- 模型 / 訓練 ----
LOOKBACK = 60             # 用過去 60 天的序列,預測第 61 天
TRAIN_RATIO = 0.8         # 80% 訓練、20% 測試(時間序列「依時間切」,不可打亂)
EPOCHS = 60
BATCH_SIZE = 32
LSTM_UNITS = 64
DROPOUT = 0.2
LEARNING_RATE = 1e-3
PATIENCE = 12             # EarlyStopping:驗證集連續沒進步就提早停

# ---- 路徑 ----
MODEL_DIR = "models"
DATA_DIR = "data"
