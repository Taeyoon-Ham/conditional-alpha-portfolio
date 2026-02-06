# =========================================
# src/utils/plot.py
# - Strategy vs Benchmark (Equity + Drawdown) in ONE figure
# =========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _equity_from_returns(r: pd.Series) -> pd.Series:
    r = r.dropna()
    eq = (1.0 + r).cumprod()
    eq.name = "equity"
    return eq


def _drawdown(eq: pd.Series) -> pd.Series:
    peak = eq.cummax()
    dd = eq / peak - 1.0
    dd.name = "drawdown"
    return dd


def plot_strategy_vs_benchmark(
    strategy_ret: pd.Series,
    bench_ret: pd.Series,
    title_prefix: str = "Strategy vs Benchmark",
    show: bool = True,
    save_path: str | None = None,
):
    # align
    idx = strategy_ret.dropna().index.intersection(bench_ret.dropna().index)
    s = strategy_ret.reindex(idx).dropna()
    b = bench_ret.reindex(idx).dropna()
    idx = s.index.intersection(b.index)
    s = s.reindex(idx)
    b = b.reindex(idx)

    s_eq = _equity_from_returns(s)
    b_eq = _equity_from_returns(b)
    s_dd = _drawdown(s_eq)
    b_dd = _drawdown(b_eq)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(s_eq.index, s_eq.values, label="Strategy")
    axes[0].plot(b_eq.index, b_eq.values, label="Benchmark")
    axes[0].set_title(f"{title_prefix} - Equity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(s_dd.index, s_dd.values, label="Strategy")
    axes[1].plot(b_dd.index, b_dd.values, label="Benchmark")
    axes[1].set_title(f"{title_prefix} - Drawdown")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig
