"""簡易回測引擎:用模型的「隔日報酬預測」當訊號,模擬交易並與買入持有比較。

策略:預測隔日報酬 > 門檻 → 做多(持有一天);否則空手(或做空)。
每次換倉扣交易成本。所有運算只用「樣本外」的測試段,避免看到未來。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(pred_df: pd.DataFrame, threshold: float = 0.0,
                 cost: float = 0.001, allow_short: bool = False):
    """pred_df 需含 prev_close / pred_close / actual_close(來自 StockModel.predict_series)。

    回傳 (daily, metrics):
      daily   — 每日持倉、策略/大盤報酬、權益曲線
      metrics — 總報酬、年化、Sharpe、最大回撤、勝率、交易次數…
    """
    pred_ret = pred_df["pred_close"] / pred_df["prev_close"] - 1.0
    actual_ret = pred_df["actual_close"] / pred_df["prev_close"] - 1.0

    long_flat = 1.0
    other = -1.0 if allow_short else 0.0
    position = pd.Series(np.where(pred_ret > threshold, long_flat, other),
                         index=pred_df.index)

    # 持倉變動量 → 交易成本(第一天建倉也算)
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])

    strat_ret = position * actual_ret - turnover * cost
    bh_ret = actual_ret

    equity = (1 + strat_ret).cumprod()
    bh_equity = (1 + bh_ret).cumprod()

    daily = pd.DataFrame({
        "position": position, "strat_ret": strat_ret, "bh_ret": bh_ret,
        "equity": equity, "bh_equity": bh_equity,
    })
    return daily, _metrics(daily)


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def _ann_return(equity: pd.Series, n: int) -> float:
    return float(equity.iloc[-1] ** (252 / n) - 1) if n > 0 else 0.0


def _sharpe(ret: pd.Series) -> float:
    s = ret.std()
    return float(ret.mean() / s * np.sqrt(252)) if s > 0 else 0.0


def _metrics(daily: pd.DataFrame) -> dict:
    strat_ret, bh_ret = daily["strat_ret"], daily["bh_ret"]
    equity, bh_equity = daily["equity"], daily["bh_equity"]
    position = daily["position"]
    n = len(daily)
    active = position != 0
    return {
        "strat_total": float(equity.iloc[-1] - 1),
        "bh_total": float(bh_equity.iloc[-1] - 1),
        "strat_ann": _ann_return(equity, n),
        "bh_ann": _ann_return(bh_equity, n),
        "strat_sharpe": _sharpe(strat_ret),
        "bh_sharpe": _sharpe(bh_ret),
        "strat_mdd": _max_drawdown(equity),
        "bh_mdd": _max_drawdown(bh_equity),
        "win_rate": float((strat_ret[active] > 0).mean()) if active.any() else 0.0,
        "n_trades": int((position.diff().abs() > 0).sum()),
        "exposure": float(active.mean()),
        "days": n,
    }
