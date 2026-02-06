import pandas as pd

def weights_from_quality(q: pd.DataFrame, gate: pd.DataFrame, top_n: int = 2) -> pd.DataFrame:
    score = q.where(gate > 0)
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)

    for dt in score.index:
        s = score.loc[dt].dropna()
        if len(s) == 0:
            continue
        pick = s.sort_values(ascending=False).head(top_n).index
        w.loc[dt, pick] = 1.0 / len(pick)

    return w
import pandas as pd

def apply_rebalance_calendar(w: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    """
    Keep weights only on rebalance dates, then forward-fill.
    freq examples:
      - "W-FRI"  (weekly Friday)
      - "ME"     (month-end)
    """
    rb = w.resample(freq).last()
    rb = rb.reindex(w.index).ffill().fillna(0.0)
    return rb
