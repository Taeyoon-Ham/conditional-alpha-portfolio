import pandas as pd

def momentum_gate(prices: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    # month-end prices
    me = prices.resample("ME").last()
    mom = me / me.shift(k) - 1.0

    # daily aligned
    mom = mom.reindex(prices.index).ffill()

    # gate: 1 if momentum > 0 else 0
    return (mom > 0).astype(float)
