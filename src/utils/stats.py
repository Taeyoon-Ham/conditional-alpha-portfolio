import numpy as np
import pandas as pd

def zscore_ts(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / sd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
