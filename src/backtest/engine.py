import pandas as pd

def run_backtest(
    rets: pd.DataFrame,
    weights: pd.DataFrame,
    cash_ret: float = 0.0,
    cost_bps: float = 5.0,  # 5bp per 100% turnover
) -> pd.DataFrame:
    w = weights.reindex(rets.index).fillna(0.0)
    cash_w = (1.0 - w.sum(axis=1)).clip(lower=0.0, upper=1.0)

    # turnover: sum(|Δw|)
    dw = w.diff().abs().sum(axis=1)
    cost = (cost_bps / 1e4) * dw

    port_ret = (w.shift(1) * rets).sum(axis=1) + cash_w.shift(1) * cash_ret - cost
    equity = (1.0 + port_ret.fillna(0.0)).cumprod()

    out = pd.DataFrame({
        "port_ret": port_ret,
        "equity": equity,
        "cash_w": cash_w,
        "turnover": dw,
        "cost": cost,
    }, index=rets.index)

    return out
