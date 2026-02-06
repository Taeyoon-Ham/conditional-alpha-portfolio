import numpy as np
import pandas as pd

from src.utils.stats import zscore_ts, sigmoid


def _ensure_df(x, index=None, columns=None) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        x = x.to_frame()
    x = x.copy()
    if index is not None:
        x = x.reindex(index)
    if columns is not None:
        x = x.reindex(columns=columns)
    # 강제 numeric (문자열 섞여도 NaN으로 처리)
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def crowding_score(
    rets: pd.DataFrame,
    dvol: pd.DataFrame,
    market_ret: pd.Series,
    vol_window: int = 21,
    corr_window: int = 63,
    z_window: int = 252,
    wL: float = 0.2,
    wV: float = 0.3,
    wC: float = 0.5,
) -> pd.Series:
    """
    CrowdingScore_t = sigmoid( wL * Z(L_t) + wV * Z(V_t) + wC * Z(C_t) )

    Robustness:
    - rets/dvol numeric 강제
    - dvol <= 0 또는 inf 방지
    - window 초기 NaN 구간은 허용하되, 전체가 NaN이 되지 않도록 보호
    """

    if rets is None or len(rets) == 0:
        return pd.Series(dtype=float)

    idx = rets.index
    cols = rets.columns

    rets = _ensure_df(rets, index=idx, columns=cols)
    dvol = _ensure_df(dvol, index=idx, columns=cols)

    # market_ret numeric + index align
    market_ret = pd.to_numeric(pd.Series(market_ret, index=idx), errors="coerce")

    # -------------------------
    # 1) Liquidity stress: L_t = mean_i(|r_i,t| / DollarVolume_i,t)
    # -------------------------
    dv = dvol.replace([np.inf, -np.inf], np.nan)
    dv = dv.mask(dv <= 0.0, np.nan)

    L = (rets.abs() / dv).mean(axis=1)
    # 너무 심하게 NaN이면(예: dvol 거의 다 NaN) -> 완화용 대체
    if L.notna().sum() < max(10, int(0.05 * len(L))):
        # dvol이 비정상이라 판단: L을 abs(rets) 평균으로 대체(최후의 방어)
        L = rets.abs().mean(axis=1)

    Lz = zscore_ts(L, z_window)

    # -------------------------
    # 2) Volatility stress: V_t = Z(std(market_ret))
    # -------------------------
    mkt_vol = market_ret.rolling(vol_window, min_periods=max(5, vol_window // 2)).std(ddof=0)
    Vz = zscore_ts(mkt_vol, z_window)

    # -------------------------
    # 3) Correlation stress: C_t = avg pairwise corr over window
    # -------------------------
    C_list = []
    for t in range(len(idx)):
        if t < corr_window - 1:
            C_list.append(np.nan)
            continue

        win = rets.iloc[t - corr_window + 1 : t + 1]
        # 완전 NaN 컬럼 제거
        win = win.dropna(axis=1, how="all")
        if win.shape[1] <= 1:
            C_list.append(np.nan)
            continue

        corr = win.corr(min_periods=max(10, corr_window // 3))
        n = corr.shape[0]
        if n <= 1:
            C_list.append(np.nan)
        else:
            off = corr.values[np.triu_indices(n, k=1)]
            C_list.append(np.nanmean(off))

    C = pd.Series(C_list, index=idx)
    Cz = zscore_ts(C, z_window)

    # -------------------------
    # Combine + sigmoid
    # -------------------------
    raw = wL * Lz + wV * Vz + wC * Cz

    # 여기서도 전부 NaN이면 전략이 죽습니다. 마지막 방어:
    if raw.notna().sum() == 0:
        # 0으로 두면 sigmoid(0)=0.5가 되어 "중립" crowding으로 동작
        raw = pd.Series(0.0, index=idx)

    score = sigmoid(raw)

    # score가 NaN이면 0.5로(중립)
    score = score.astype(float).fillna(0.5)
    score.name = "CrowdingScore"
    return score
