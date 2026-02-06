import pandas as pd

def amplification(crowding: pd.Series, A_min: float = 0.2, A_max: float = 1.0) -> pd.Series:
    return A_min + (A_max - A_min) * (1 - crowding)
