"""LSTM 股價預測模型。

核心做法:**不直接預測絕對價格**,而是預測「隔日 log 報酬率」,再用
    price_t = close_{t-1} * exp(預測報酬)
還原成價格。報酬率近似平穩,縮放不會飽和,對台積電這種趨勢股表現遠優於直接預測價格。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from src.features import build_dataset

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def create_sequences(features: np.ndarray, target: np.ndarray, lookback: int):
    """滑動視窗:用第 [t-lookback, t-1] 天的特徵,預測第 t 天的目標(當日報酬)。

    回傳 X(樣本數, lookback, 特徵數)、y(樣本數,)、以及每個樣本對應的原始列索引 t。
    """
    X, y, idx = [], [], []
    for t in range(lookback, len(features)):
        X.append(features[t - lookback:t])
        y.append(target[t])
        idx.append(t)
    return np.array(X), np.array(y), np.array(idx)


def build_lstm(input_shape) -> Sequential:
    """兩層 LSTM + Dropout 的回歸模型。"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(config.LSTM_UNITS, return_sequences=True),
        Dropout(config.DROPOUT),
        LSTM(config.LSTM_UNITS // 2),
        Dropout(config.DROPOUT),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    # Huber loss 對極端報酬日較穩健(不會被少數大漲大跌過度影響)
    model.compile(optimizer=Adam(config.LEARNING_RATE), loss="huber", metrics=["mae"])
    return model


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, prev_close: np.ndarray) -> dict:
    """回歸誤差 + 方向性指標(以前一日真實收盤為基準判斷漲跌)。"""
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / y_true)) * 100)
    dir_true = np.sign(y_true - prev_close)
    dir_pred = np.sign(y_pred - prev_close)
    direction = float(np.mean(dir_true == dir_pred) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "DirectionAcc": direction}


@dataclass
class StockModel:
    """封裝 scaler + LSTM,提供 fit / evaluate / forecast / save / load。"""

    ticker: str = config.TICKER
    lookback: int = config.LOOKBACK
    feature_scaler: StandardScaler = field(default_factory=StandardScaler)
    target_scaler: StandardScaler = field(default_factory=StandardScaler)
    model: Sequential | None = None
    features: list = field(default_factory=lambda: list(config.FEATURES))  # 此模型實際用到的特徵

    # 訓練後保留的測試結果(供畫圖)
    test_dates: pd.DatetimeIndex | None = None
    test_true: np.ndarray | None = None
    test_pred: np.ndarray | None = None
    history: dict | None = None
    metrics: dict | None = None

    # ---------- 訓練 ----------
    def fit(self, df: pd.DataFrame, epochs: int = config.EPOCHS,
            batch_size: int = config.BATCH_SIZE, extra_callbacks=None, verbose: int = 1):
        """df 為原始 OHLCV。特徵工程(技術+法人)-> 三段切分 -> 縮放 -> 建序列 -> 訓練 -> 測試指標。"""
        feat, self.features = build_dataset(df, self.ticker)
        n = len(feat)
        train_end = int(n * config.TRAIN_RATIO)   # 訓練段結束
        test_start = _test_start(n)               # 測試段開始(中間為驗證段)

        close = feat["Close"].values
        # 目標 = 當日 log 報酬(= log(close_t / close_{t-1}));feat["Return"] 已是此值
        target = feat["Return"].values

        # scaler 只用「訓練段」fit,避免資訊洩漏(look-ahead bias)
        self.feature_scaler.fit(feat[self.features].iloc[:train_end])
        self.target_scaler.fit(target[:train_end].reshape(-1, 1))

        scaled_x = self.feature_scaler.transform(feat[self.features])
        scaled_y = self.target_scaler.transform(target.reshape(-1, 1)).ravel()

        X, y, idx = create_sequences(scaled_x, scaled_y, self.lookback)
        train_mask = idx < train_end
        val_mask = (idx >= train_end) & (idx < test_start)
        test_mask = idx >= test_start

        self.model = build_lstm((self.lookback, len(self.features)))
        callbacks = [EarlyStopping(monitor="val_loss", patience=config.PATIENCE,
                                   restore_best_weights=True)]
        if extra_callbacks:
            callbacks += extra_callbacks

        # 早停看「驗證集」,完全不碰測試集 → 測試指標才是誠實的樣本外表現
        hist = self.model.fit(
            X[train_mask], y[train_mask],
            validation_data=(X[val_mask], y[val_mask]),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=verbose,
        )
        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}

        # 把測試集的「報酬預測」還原成價格:price_t = close_{t-1} * exp(pred_return_t)
        test_idx = idx[test_mask]
        pred_ret = self.target_scaler.inverse_transform(
            self.model.predict(X[test_mask], verbose=0)).ravel()
        prev_close = close[test_idx - 1]
        self.test_pred = prev_close * np.exp(pred_ret)
        self.test_true = close[test_idx]
        self.test_dates = feat.index[test_idx]
        self.metrics = _metrics(self.test_true, self.test_pred, prev_close)
        return self

    # ---------- 未來預測(遞迴) ----------
    def forecast(self, df: pd.DataFrame, days: int = 5) -> pd.DataFrame:
        """從最後一天往後遞迴預測 days 天。

        每預測一天報酬 -> 還原收盤價 -> 接回序列 -> 重算技術指標 -> 預測下一天。
        未知的 Open/High/Low 以預測收盤近似、Volume 沿用最後一日。
        回傳 DataFrame(index=未來日期, 欄位 PredictedClose)。
        """
        work = df.copy()
        preds, dates = [], []
        # 未來不知道盤中高低與成交量,用近 20 日的平均高低振幅與均量近似,
        # 讓 ATR / 振幅 / 量能等特徵在遞迴預測時仍落在合理範圍。
        recent = work.tail(20)
        half_range = float(((recent["High"] - recent["Low"]) / recent["Close"]).mean()) / 2
        avg_volume = float(recent["Volume"].mean())

        for _ in range(days):
            feat, _ = build_dataset(work, self.ticker)
            feat = self._select_features(feat)
            window = self.feature_scaler.transform(feat.iloc[-self.lookback:])
            pred_ret = float(self.target_scaler.inverse_transform(
                self.model.predict(window[np.newaxis, ...], verbose=0))[0, 0])

            last_close = float(work["Close"].iloc[-1])
            pred_close = last_close * np.exp(pred_ret)
            next_date = _next_business_day(work.index[-1])
            preds.append(pred_close)
            dates.append(next_date)

            work.loc[next_date] = {
                "Open": last_close,
                "High": pred_close * (1 + half_range),
                "Low": pred_close * (1 - half_range),
                "Close": pred_close,
                "Volume": avg_volume,
            }

        return pd.DataFrame({"PredictedClose": preds}, index=pd.DatetimeIndex(dates, name="Date"))

    # ---------- 逐日預測(供回測) ----------
    def predict_series(self, df: pd.DataFrame, only_test: bool = True) -> pd.DataFrame:
        """對 df 的每一天跑出隔日預測,回傳 DataFrame(index=日期),欄位:
        prev_close(前一日收盤)、pred_close(模型預測收盤)、actual_close(實際收盤)。
        only_test=True 時只取「測試段(樣本外)」,與訓練時的切分一致,供回測使用。
        """
        feat, _ = build_dataset(df, self.ticker)
        test_start = _test_start(len(feat))
        close = feat["Close"].values

        scaled_x = self.feature_scaler.transform(self._select_features(feat))
        X, _dummy, idx = create_sequences(scaled_x, close, self.lookback)
        pred_ret = self.target_scaler.inverse_transform(
            self.model.predict(X, verbose=0)).ravel()

        prev_close = close[idx - 1]
        out = pd.DataFrame(
            {
                "prev_close": prev_close,
                "pred_close": prev_close * np.exp(pred_ret),
                "actual_close": close[idx],
            },
            index=feat.index[idx],
        )
        if only_test:
            out = out[idx >= test_start]
        return out

    def _select_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        """取出本模型訓練時用到的特徵欄位;若某欄缺失(例如法人資料臨時抓不到)以 0 補,
        確保特徵數與訓練時一致,推論不會因此失敗(法人特徵=0 即視為中性)。"""
        for col in self.features:
            if col not in feat.columns:
                feat[col] = 0.0
        return feat[self.features]

    # ---------- 存 / 讀 ----------
    def save(self, model_dir: str = config.MODEL_DIR):
        os.makedirs(model_dir, exist_ok=True)
        tag = self.ticker.replace(".", "_")
        self.model.save(os.path.join(model_dir, f"{tag}.keras"))
        joblib.dump(
            {
                "ticker": self.ticker,
                "lookback": self.lookback,
                "features": self.features,
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
                "metrics": self.metrics,
                "history": self.history,
                # 一併保存測試集對照,載入舊模型時也能畫出完整預測圖
                "test_dates": self.test_dates,
                "test_true": self.test_true,
                "test_pred": self.test_pred,
            },
            os.path.join(model_dir, f"{tag}_meta.pkl"),
        )

    @classmethod
    def load(cls, ticker: str = config.TICKER, model_dir: str = config.MODEL_DIR) -> "StockModel":
        tag = ticker.replace(".", "_")
        meta = joblib.load(os.path.join(model_dir, f"{tag}_meta.pkl"))
        obj = cls(
            ticker=meta["ticker"],
            lookback=meta["lookback"],
            features=meta.get("features", list(config.FEATURES)),
            feature_scaler=meta["feature_scaler"],
            target_scaler=meta["target_scaler"],
        )
        obj.model = tf.keras.models.load_model(os.path.join(model_dir, f"{tag}.keras"))
        obj.metrics = meta.get("metrics")
        obj.history = meta.get("history")
        obj.test_dates = meta.get("test_dates")
        obj.test_true = meta.get("test_true")
        obj.test_pred = meta.get("test_pred")
        return obj

    @staticmethod
    def exists(ticker: str = config.TICKER, model_dir: str = config.MODEL_DIR) -> bool:
        tag = ticker.replace(".", "_")
        return os.path.exists(os.path.join(model_dir, f"{tag}.keras"))


def _test_start(n: int) -> int:
    """測試段(樣本外)起始索引 = 訓練段 + 驗證段之後。"""
    return int(n * (config.TRAIN_RATIO + config.VAL_RATIO))


def _next_business_day(date) -> pd.Timestamp:
    """下一個工作日(週末跳過;台股國定假日未細算,作為示範足夠)。"""
    return pd.Timestamp(date) + pd.offsets.BDay(1)
