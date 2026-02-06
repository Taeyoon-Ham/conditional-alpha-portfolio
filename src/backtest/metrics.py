import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def worst_q_return(r: pd.Series, q: float = 0.05) -> float:
    return float(r.quantile(q))

def sharpe(r: pd.Series, ann: int = 252) -> float:
    r = r.dropna()
    if r.std(ddof=0) == 0:
        return float("nan")
    return float((r.mean() / r.std(ddof=0)) * np.sqrt(ann))

def conditional_report(
    port_ret: pd.Series,
    equity: pd.Series,
    crowding: pd.Series,
    split_q: float = 0.8,   # top 20% crowding = high
) -> pd.DataFrame:
    c = crowding.reindex(port_ret.index)
    hi = c >= c.quantile(split_q)
    lo = c <= c.quantile(1 - split_q)

    def pack(mask: pd.Series) -> dict:
        rr = port_ret[mask].dropna()
        eq = (1.0 + rr).cumprod()
        return {
            "n_days": int(rr.shape[0]),
            "mean": float(rr.mean()),
            "vol": float(rr.std(ddof=0)),
            "sharpe": sharpe(rr),
            "worst_5%": worst_q_return(rr, 0.05),
            "mdd": max_drawdown(eq),
            "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else float("nan"),
        }

    out = pd.DataFrame({
        "ALL": pack(port_ret.notna()),
        "LOW_CROWD": pack(lo),
        "HIGH_CROWD": pack(hi),
    }).T

    return out
