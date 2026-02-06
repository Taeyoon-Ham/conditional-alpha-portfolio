# =========================================
# src/main.py
# FIX 2 problems:
# (1) Too many train candidates -> remove grid/WF
#     - Random search with hard cap + early stop
#     - SPEED: precompute signals once + cache crowd thresholds
# (2) TEST underperforms SPY -> SPY core + gated small overlay
#     - Overlay turns ON only if recent active IR > 0 (lookback 252d)
#     - Weekly rebalance to cut turnover
#     - FIX: weekly_rebalance bug + use abs() gross caps
# =========================================

from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.data.loader import load_universe
from src.signals.crowding import crowding_score
from src.backtest.engine import run_backtest
from src.backtest.metrics import conditional_report
from src.utils.plot import plot_strategy_vs_benchmark


# -----------------------------
# Config (speed hard caps)
# -----------------------------
START = "2010-01-01"
TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
COST_BPS = 10.0

# SPEED: hard limits
N_CANDIDATES = 200             # absolute cap
EARLY_STOP_PATIENCE = 60       # stop if no improvement for N candidates
PROGRESS_EVERY_SEC = 2.0

# HOLDOUT TEST
TEST_FRAC = 0.20

# Overlay gate
GATE_LOOKBACK = 252            # 1Y
GATE_MIN_IR = 0.00             # ON only if recent 1Y active IR > 0

# Rebalance
REBALANCE_N = 5                # weekly


# -----------------------------
# Helpers
# -----------------------------
def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing tickers: {missing}. Loaded: {list(df.columns)}")


def ann_stats(ret: pd.Series):
    r = ret.dropna()
    if len(r) < 3:
        return (np.nan, np.nan, np.nan, np.nan)
    eq = (1 + r).cumprod()
    yrs = len(r) / 252.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(252)
    mdd = (eq / eq.cummax() - 1).min()
    return float(cagr), float(vol), float(sharpe), float(mdd)


def total_return(ret: pd.Series) -> float:
    r = ret.dropna()
    if len(r) == 0:
        return float("nan")
    return float((1 + r).prod() - 1.0)


def ols_alpha_beta(strategy_ret: pd.Series, spy_ret: pd.Series):
    s = strategy_ret.dropna()
    b = spy_ret.reindex(s.index).dropna()
    s = s.reindex(b.index)
    if len(s) < 120:
        return (np.nan, np.nan, np.nan, np.nan)
    x = b.values
    y = s.values
    xm, ym = x.mean(), y.mean()
    cov = np.mean((x - xm) * (y - ym))
    var = np.mean((x - xm) ** 2) + 1e-12
    beta = cov / var
    alpha_d = ym - beta * xm
    resid = y - (alpha_d + beta * x)
    te_ann = np.std(resid, ddof=1) * np.sqrt(252)
    alpha_ann = alpha_d * 252
    ir = alpha_ann / (te_ann + 1e-12)
    return float(alpha_ann), float(beta), float(te_ann), float(ir)


def weekly_rebalance(w: pd.DataFrame, every_n: int = 5) -> pd.DataFrame:
    """
    Keep weights only every_n rows, forward-fill between rebalances.
    FIX: use iloc-based masking (previous boolean-index approach could break subtly).
    """
    if len(w) == 0:
        return w
    out = w.copy()
    keep = np.zeros(len(out), dtype=bool)
    keep[::every_n] = True
    out.iloc[~keep, :] = np.nan
    return out.ffill().fillna(0.0)


def active_ir_rolling(active: pd.Series, lookback: int = 252) -> pd.Series:
    mu = active.rolling(lookback).mean()
    sd = active.rolling(lookback).std(ddof=1)
    ir = (mu / (sd + 1e-12)) * np.sqrt(252.0)
    return ir


# -----------------------------
# Precomputed signals (speed)
# -----------------------------
@dataclass(frozen=True)
class Signals:
    px: pd.DataFrame
    spy_px: pd.Series
    tlt_ok: pd.Series
    crowd: pd.Series


# -----------------------------
# Signals / Weights
# -----------------------------
def prepare_signals(rets: pd.DataFrame, dvol: pd.DataFrame) -> Signals:
    _require_cols(rets, TICKERS)
    px = (1 + rets.fillna(0.0)).cumprod()
    spy_px = px["SPY"]
    tlt_px = px["TLT"]
    tlt_ma = tlt_px.rolling(126).mean()

    mkt_ret = rets.mean(axis=1)
    crowd = crowding_score(rets, dvol, mkt_ret).rename("CrowdingScore")
    tlt_ok = (tlt_px > tlt_ma).rename("TLT_OK")

    return Signals(px=px, spy_px=spy_px, tlt_ok=tlt_ok, crowd=crowd)


def build_overlay(rets: pd.DataFrame, sig: Signals, p: dict, crowd_thr: float) -> pd.DataFrame:
    """
    Build alpha overlay (no SPY core here).
    Overlay is small, regime-based:
      - BullStrong: QQQ/IWM tilt
      - Bear + TLT OK: TLT/GLD
      - Otherwise: 0
    Crowding filter: overlay only when crowd <= precomputed threshold (crowd_thr)
    """
    idx = rets.index
    spy_px = sig.spy_px
    tlt_ok = sig.tlt_ok

    ma_fast = int(p["ma_fast"])
    ma_slow = int(p["ma_slow"])
    mom_lb = int(p["mom_lb"])

    ma_f = spy_px.rolling(ma_fast).mean()
    ma_s = spy_px.rolling(ma_slow).mean()
    bull = ma_f > ma_s
    bear = ~bull

    mom = spy_px / spy_px.shift(mom_lb) - 1.0
    bull_strong = bull & (mom > 0.0)

    crowd = sig.crowd.reindex(idx)
    thr = float(crowd_thr)
    low_crowd = crowd <= thr

    w = pd.DataFrame(0.0, index=idx, columns=rets.columns)

    # BullStrong overlay
    qqq_w = float(p["qqq_w"])
    iwm_w = 1.0 - qqq_w
    w.loc[low_crowd & bull_strong, "QQQ"] = qqq_w
    w.loc[low_crowd & bull_strong, "IWM"] = iwm_w

    # Bear defensive overlay (only if TLT trend ok)
    w.loc[low_crowd & bear & tlt_ok, "TLT"] = float(p["bear_tlt_w"])
    w.loc[low_crowd & bear & tlt_ok, "GLD"] = float(p["bear_gld_w"])

    # weekly rebalance to cut turnover
    w = weekly_rebalance(w, every_n=REBALANCE_N)

    # overlay gross cap (use abs to be robust)
    gross = w.abs().sum(axis=1)
    cap = float(p["overlay_gross_cap"])
    over = gross > cap
    if over.any():
        w.loc[over] = w.loc[over].mul(cap / gross.loc[over], axis=0)

    return w


def combine_core_and_overlay(rets: pd.DataFrame, spy_ret: pd.Series, overlay: pd.DataFrame, p: dict) -> pd.DataFrame:
    """
    Total weights = SPY core + overlay_k * overlay
    Overlay is gated by recent active IR (pre-cost proxy):
      - If recent IR <= threshold -> overlay_k = 0
    """
    idx = rets.index
    w = pd.DataFrame(0.0, index=idx, columns=rets.columns)
    w["SPY"] = 1.0  # core

    # proxy for active = overlay sleeve return (since core=SPY, active comes from overlay)
    overlay_ret_proxy = (overlay * rets).sum(axis=1)
    ir = active_ir_rolling(overlay_ret_proxy, lookback=GATE_LOOKBACK)

    k = float(p["overlay_k"])
    gate = ir > float(p["gate_min_ir"])
    k_t = pd.Series(0.0, index=idx)
    k_t.loc[gate] = k

    w = w.add(overlay.mul(k_t, axis=0), fill_value=0.0)

    # total leverage cap (use abs to be robust)
    lev_cap = float(p["lev_cap"])
    gross = w.abs().sum(axis=1)
    over = gross > lev_cap
    if over.any():
        w.loc[over] = w.loc[over].mul(lev_cap / gross.loc[over], axis=0)

    return w


def eval_params(
    rets: pd.DataFrame,
    dvol: pd.DataFrame,
    spy_ret: pd.Series,
    p: dict,
    sig: Signals,
    crowd_thr: float,
) -> dict:
    overlay = build_overlay(rets, sig, p, crowd_thr=crowd_thr)
    w = combine_core_and_overlay(rets, spy_ret, overlay, p)

    bt = run_backtest(rets=rets, weights=w, cash_ret=0.0, cost_bps=COST_BPS)
    strat = bt["port_ret"].dropna()
    spy = spy_ret.reindex(strat.index).dropna()
    strat = strat.reindex(spy.index)

    strat_total = total_return(strat)
    spy_total = total_return(spy)
    excess = float(strat_total - spy_total)

    alpha_ann, beta, te_ann, ir = ols_alpha_beta(strat, spy)
    active = strat - spy
    active_mean_ann = float(active.mean() * 252.0) if len(active) else float("nan")
    t_active = (
        float(active.mean() / ((active.std(ddof=1) + 1e-12) / np.sqrt(len(active))))
        if len(active) > 30
        else float("nan")
    )

    cagr, vol, sharpe, mdd = ann_stats(strat)

    avg_turn = float(bt["turnover"].dropna().mean()) if "turnover" in bt else float("nan")
    gross_max = float(w.abs().sum(axis=1).max())

    return {
        "p": p,
        "bt": bt,
        "weights": w,
        "overlay": overlay,
        "excess": excess,
        "alpha_ann": alpha_ann,
        "beta": beta,
        "ir": ir,
        "active_mean_ann": active_mean_ann,
        "t_active": t_active,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "avg_turn": avg_turn,
        "gross_max": gross_max,
        "spy_total": spy_total,
        "str_total": strat_total,
        "crowd": sig.crowd,
    }


def score_selection(res: dict) -> float:
    """
    Selection objective:
      - Prefer positive active_mean_ann and alpha_ann
      - Keep beta close to 1
      - Penalize drawdown and turnover
    """
    if np.isnan(res["alpha_ann"]) or np.isnan(res["beta"]) or np.isnan(res["mdd"]):
        return -1e18

    s = 0.0
    s += 1.20 * res["active_mean_ann"]
    s += 0.20 * res["alpha_ann"]
    s += 0.10 * (res["ir"] if not np.isnan(res["ir"]) else 0.0)
    s -= 0.25 * abs(res["beta"] - 1.0)
    s -= 0.15 * abs(res["mdd"])
    if not np.isnan(res["avg_turn"]):
        s -= 0.20 * res["avg_turn"]
    return float(s)


# -----------------------------
# Random search (fast, capped)
# -----------------------------
def sample_params(rng: np.random.Generator) -> dict:
    # Narrow, stable ranges (avoid huge search space)
    ma_fast = int(rng.choice([90, 110, 126]))
    ma_slow = int(rng.choice([180, 220, 252]))
    if ma_fast >= ma_slow:
        ma_fast, ma_slow = 110, 220

    return {
        "ma_fast": ma_fast,
        "ma_slow": ma_slow,
        "mom_lb": 126,
        "qqq_w": float(rng.choice([0.85, 0.90])),
        "bear_tlt_w": float(rng.choice([0.35, 0.45, 0.55])),
        "bear_gld_w": float(rng.choice([0.15, 0.25, 0.35])),
        # keep crowd_q not too high (sample stability)
        "crowd_q": float(rng.choice([0.80, 0.85, 0.90])),
        # overlay sizing (small)
        "overlay_k": float(rng.choice([0.25, 0.35, 0.45])),
        "overlay_gross_cap": 1.0,
        "lev_cap": float(rng.choice([1.10, 1.20])),
        # gating threshold (fixed)
        "gate_min_ir": GATE_MIN_IR,
    }


def search_best(rets_sel, dvol_sel, spy_sel, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    # SPEED: precompute signals once (selection window)
    sig_sel = prepare_signals(rets_sel, dvol_sel)

    # SPEED: cache crowd thresholds for allowed crowd_q values
    crowd_q_values = [0.80, 0.85, 0.90]
    crowd_thr_map = {q: float(sig_sel.crowd.quantile(q)) for q in crowd_q_values}

    best = None
    best_score = -1e18
    no_improve = 0

    t0 = time.time()
    last = time.time()

    print("\n=== SEARCH START (TRAIN/SELECTION) ===")
    for i in range(1, N_CANDIDATES + 1):
        p = sample_params(rng)
        cq = float(p["crowd_q"])
        thr = crowd_thr_map.get(cq, float(sig_sel.crowd.quantile(cq)))

        res = eval_params(rets_sel, dvol_sel, spy_sel, p, sig=sig_sel, crowd_thr=thr)
        sc = score_selection(res)

        if sc > best_score:
            best_score = sc
            best = res
            no_improve = 0
        else:
            no_improve += 1

        now = time.time()
        if now - last >= PROGRESS_EVERY_SEC:
            pct = (i / N_CANDIDATES) * 100.0
            print(f"[TRAIN] {i}/{N_CANDIDATES} ({pct:.1f}%)  best_score={best_score:.4f}  elapsed={now - t0:.1f}s")
            last = now

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"[EARLY STOP] no improvement for {EARLY_STOP_PATIENCE} candidates.")
            break

    if best is None:
        raise RuntimeError("Search failed: no candidate evaluated.")
    return best


# -----------------------------
# Main
# -----------------------------
def main():
    print("Main start")

    rets, dvol, spy_ret = load_universe(TICKERS, START, end=None, use_cache=True)
    common = rets.index.intersection(spy_ret.index)
    rets = rets.reindex(common).dropna(how="all")
    dvol = dvol.reindex(common)
    spy_ret = spy_ret.reindex(common).dropna()
    _require_cols(rets, TICKERS)

    # split
    n = len(rets)
    cut = int(n * (1.0 - TEST_FRAC))
    rets_sel, rets_te = rets.iloc[:cut], rets.iloc[cut:]
    dvol_sel, dvol_te = dvol.iloc[:cut], dvol.iloc[cut:]
    spy_sel, spy_te = spy_ret.iloc[:cut], spy_ret.iloc[cut:]

    # search (FAST)
    best = search_best(rets_sel, dvol_sel, spy_sel, seed=42)
    pbest = best["p"]

    print("\n=== BEST PARAMS (FAST SEARCH) ===")
    for k, v in pbest.items():
        print(f"{k}: {v}")

    # EVAL SELECTION
    print("\n=== EVAL ON SELECTION (TRAIN) ===")
    res_sel = best
    print("SEL Excess vs SPY:", res_sel["excess"])
    print("SEL Active_mean_ann:", res_sel["active_mean_ann"])
    print("SEL Alpha_ann:", res_sel["alpha_ann"], " Beta:", res_sel["beta"], " IR:", res_sel["ir"])
    print("SEL mdd:", res_sel["mdd"], " gross_max:", res_sel["gross_max"], " avg_turn:", res_sel["avg_turn"])

    # EVAL TEST
    print("\n=== EVAL ON HOLDOUT TEST ===")
    sig_te = prepare_signals(rets_te, dvol_te)
    thr_te = float(sig_te.crowd.quantile(float(pbest["crowd_q"])))
    res_te = eval_params(rets_te, dvol_te, spy_te, pbest, sig=sig_te, crowd_thr=thr_te)

    print("\n===== Strategy vs SPY (TEST) =====")
    print("SPY total return:", res_te["spy_total"])
    print("Strategy total return:", res_te["str_total"])
    print("Excess vs SPY:", res_te["excess"])
    print("Active_mean_ann:", res_te["active_mean_ann"])
    print("Alpha_ann (OLS vs SPY):", res_te["alpha_ann"])
    print("Beta (OLS vs SPY):", res_te["beta"])
    print("IR:", res_te["ir"])
    print("t-stat(active):", res_te["t_active"])
    print("mdd:", res_te["mdd"], " gross_max:", res_te["gross_max"], " avg_turn:", res_te["avg_turn"])

    print("\n[Annualized TEST]")
    print("SPY :", ann_stats(spy_te))
    print("STR :", (res_te["cagr"], res_te["vol"], res_te["sharpe"], res_te["mdd"]))

    print("\nConditional report (TEST)")
    crowd_te = res_te["crowd"].reindex(res_te["bt"].index)
    print(conditional_report(res_te["bt"]["port_ret"], res_te["bt"]["equity"], crowd_te, split_q=float(pbest["crowd_q"])))

    plot_strategy_vs_benchmark(
        strategy_ret=res_te["bt"]["port_ret"],
        bench_ret=spy_te.reindex(res_te["bt"].index),
        title_prefix="HOLDOUT TEST: SPY Core + Gated Overlay",
        show=True,
        save_path=None,
    )

    # EVAL ALL
    print("\n=== FINAL RUN (FULL SAMPLE) ===")
    sig_all = prepare_signals(rets, dvol)
    thr_all = float(sig_all.crowd.quantile(float(pbest["crowd_q"])))
    res_all = eval_params(rets, dvol, spy_ret, pbest, sig=sig_all, crowd_thr=thr_all)

    print("\n===== Strategy vs SPY (ALL) =====")
    print("SPY total return:", res_all["spy_total"])
    print("Strategy total return:", res_all["str_total"])
    print("Excess vs SPY:", res_all["excess"])
    print("Active_mean_ann:", res_all["active_mean_ann"])
    print("Alpha_ann (OLS vs SPY):", res_all["alpha_ann"])
    print("Beta (OLS vs SPY):", res_all["beta"])
    print("IR:", res_all["ir"])
    print("t-stat(active):", res_all["t_active"])
    print("mdd:", res_all["mdd"], " gross_max:", res_all["gross_max"], " avg_turn:", res_all["avg_turn"])

    print("\n[Annualized ALL]")
    print("SPY :", ann_stats(spy_ret))
    print("STR :", (res_all["cagr"], res_all["vol"], res_all["sharpe"], res_all["mdd"]))

    print("\nConditional report (ALL)")
    crowd_all = res_all["crowd"].reindex(res_all["bt"].index)
    print(conditional_report(res_all["bt"]["port_ret"], res_all["bt"]["equity"], crowd_all, split_q=float(pbest["crowd_q"])))

    plot_strategy_vs_benchmark(
        strategy_ret=res_all["bt"]["port_ret"],
        bench_ret=spy_ret.reindex(res_all["bt"].index),
        title_prefix="FULL SAMPLE: SPY Core + Gated Overlay",
        show=True,
        save_path=None,
    )


if __name__ == "__main__":
    main()
