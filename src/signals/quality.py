import pandas as pd

def quality_score(rets: pd.DataFrame, vol_window: int = 21) -> pd.DataFrame:
    vol = rets.rolling(vol_window).std(ddof=0)
    mu = vol.mean(axis=1)
    sd = vol.std(axis=1, ddof=0)
    z = vol.sub(mu, axis=0).div(sd, axis=0)
    return -z
