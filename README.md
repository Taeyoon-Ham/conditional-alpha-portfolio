# Conditional Alpha Portfolio (SPY Core + Gated Overlay)

## ENGLISH (Full)

### 1) Executive Summary
This repository implements a **research-grade** backtest for a **core + overlay** portfolio:

- **Core**: 100% SPY (market beta exposure)
- **Overlay**: a small tactical sleeve across {QQQ, IWM, TLT, GLD}
- **Crowding filter**: take overlay risk only when the market is **low-crowding**
- **Gate**: overlay is **ON** only if the **recent active Information Ratio (IR)** is positive (lookback = 252 trading days)
- **Weekly rebalance**: reduce turnover and transaction-cost drag
- **Optimization**: capped random search with early stopping (no grid / no workflow explosion)

The goal is **not** to replace market exposure, but to add **conditional alpha** while retaining a robust baseline.

---

### 2) Strategy Definition

#### 2.1 Universe
- SPY: US large-cap equity (benchmark and core holding)
- QQQ: US growth/tech tilt
- IWM: US small-cap tilt
- TLT: long-duration US Treasuries (risk-off hedge)
- GLD: gold (diversifier / crisis hedge)

All assets are treated as liquid ETFs for research purposes.

#### 2.2 Core + Overlay Construction
Let:
- \( r_{t,i} \) = daily return of asset \( i \) at day \( t \)
- \( \mathbf{w}^{core}_t \) = core weights (SPY = 1.0)
- \( \mathbf{w}^{ov}_t \) = overlay weights (tactical sleeve)
- \( k_t \in \{0, k\} \) = overlay scaling factor (gated)

Total portfolio weights:
\[
\mathbf{w}_t = \mathbf{w}^{core}_t + k_t \cdot \mathbf{w}^{ov}_t
\]
where \( \mathbf{w}^{core}_t = (1,0,0,0,0) \) in the ticker order [SPY, QQQ, IWM, TLT, GLD].

Daily portfolio return:
\[
r^{port}_t = \sum_i w_{t,i} \cdot r_{t,i} - \text{costs}_t
\]

#### 2.3 Regime Logic (Overlay Rules)
The overlay is **regime-based**, using SPY trend and momentum:

- Trend filter:
  - Compute moving averages of SPY price:
    - \( MA_f(t) \) with window `ma_fast`
    - \( MA_s(t) \) with window `ma_slow`
  - Bull regime if \( MA_f(t) > MA_s(t) \); else bear regime.

- Momentum filter:
  - \( mom(t) = \frac{P_{SPY}(t)}{P_{SPY}(t - L)} - 1 \), with `mom_lb = 126`
  - BullStrong if bull regime AND \( mom(t) > 0 \)

Overlay allocation:
- **BullStrong + Low crowding**: allocate between QQQ and IWM  
  - QQQ weight = `qqq_w`, IWM weight = `1 - qqq_w`
- **Bear + Low crowding + TLT OK**: defensive overlay into TLT and GLD  
  - TLT trend-ok if \( P_{TLT}(t) > MA_{126}(TLT, t) \)
  - TLT weight = `bear_tlt_w`, GLD weight = `bear_gld_w`
- Otherwise: overlay = 0

Overlay is then:
- **Weekly rebalanced** (every 5 trading days)
- Capped by an overlay gross exposure cap

#### 2.4 Crowding Filter
A crowding score is computed from returns and dollar volume (see `src/signals/crowding.py`).  
We define a low-crowding threshold by a quantile \( q \) (parameter `crowd_q`):

\[
\text{LowCrowd}(t) \iff Crowd(t) \leq Q_q(Crowd)
\]

Overlay positions are allowed only when `LowCrowd(t)` is true.

#### 2.5 Overlay Gate (Active IR)
Overlay can harm out-of-sample if it turns into noise.  
We therefore gate overlay deployment using a recent active IR proxy.

Since core = SPY, the overlay sleeve return serves as an active proxy:
\[
r^{ov}_t = \sum_i w^{ov}_{t,i} \cdot r_{t,i}
\]

Rolling active IR over lookback \( N = 252 \):
\[
IR_t = \frac{\mu_t}{\sigma_t} \sqrt{252}
\quad\text{where}\quad
\mu_t = \text{mean}(r^{ov}_{t-N:t}),\;
\sigma_t = \text{std}(r^{ov}_{t-N:t})
\]

Gate rule:
- if \( IR_t > 0 \Rightarrow k_t = k \)
- else \( k_t = 0 \)

This makes the strategy **adaptive**: it uses overlay only when it has been recently beneficial.

#### 2.6 Risk & Exposure Caps
- Overlay gross cap (absolute weights):
\[
\sum_i |w^{ov}_{t,i}| \leq \text{overlay\_gross\_cap}
\]
- Total portfolio leverage cap:
\[
\sum_i |w_{t,i}| \leq \text{lev\_cap}
\]

---

### 3) Backtest Engine & Transaction Costs

#### 3.1 Backtest Mechanics
At each day \( t \), given weights \( \mathbf{w}_t \), portfolio return is computed by:
- Weighted asset returns
- Subtract transaction costs driven by turnover

#### 3.2 Turnover & Costs
Turnover is typically approximated by changes in weights:
\[
\text{turnover}_t \approx \sum_i |w_{t,i} - w_{t-1,i}|
\]
Transaction costs:
\[
\text{cost}_t = \text{turnover}_t \times \text{cost\_bps}
\]
where `cost_bps = 10.0` is a research assumption.

Weekly rebalance is used to reduce turnover and increase realism.

---

### 4) Experiment Design (Train/Test)

#### 4.1 Time Split
Data is split into:
- **Selection (Train)**: first \( 1 - \text{TEST\_FRAC} \) fraction
- **Holdout (Test)**: last `TEST_FRAC` fraction

Parameters are selected on Train only. Test is untouched for evaluation.

#### 4.2 Search Procedure (Capped Random Search)
To avoid excessive candidates:
- `N_CANDIDATES` = hard cap
- `EARLY_STOP_PATIENCE` = stop after no improvement for N candidates
- Parameter ranges are intentionally narrow and stable.

Objective function favors:
- positive active mean return
- positive alpha
- beta near 1
- lower drawdown and turnover

This is a pragmatic research approach intended to reduce overfitting risk.

---

### 5) Performance Reporting

#### 5.1 Key Metrics
- **Total return**: \( \prod_t (1+r_t) - 1 \)
- **CAGR**:
\[
CAGR = \left(\prod_t (1+r_t)\right)^{1/\text{years}} - 1
\]
- **Volatility (annualized)**: \( \sigma(r) \sqrt{252} \)
- **Sharpe (annualized)**:
\[
Sharpe = \frac{\mu(r)}{\sigma(r)} \sqrt{252}
\]
- **Max Drawdown (MDD)**: min of equity / running max - 1
- **OLS Alpha/Beta vs SPY**:
  - beta = covariance(strategy, SPY) / variance(SPY)
  - alpha = residual mean (annualized)
- **Information Ratio (IR)**: annualized alpha / tracking error

#### 5.2 Conditional Performance (Crowding Regimes)
The code also reports metrics separately for:
- LOW_CROWD days
- HIGH_CROWD days

This helps validate “conditional alpha” behavior:
- Does overlay work primarily in the intended regime?

---

### 6) Limitations & Recommended Further Validation

This is research code, not production trading software. Common limitations:

1) **Data & execution realism**
- Slippage is simplified
- No bid-ask modeling
- ETF liquidity is assumed

2) **Lookahead / alignment**
- Always ensure signal computation uses only information available at time \( t \)

3) **Overfitting risk**
- Even capped random search can overfit
- Parameter stability should be tested

Recommended next validations (research-grade):
- **Walk-forward analysis** (multiple train/test windows)
- **Cost stress test** (e.g., 10bps → 20bps / 30bps)
- **Bootstrap / block bootstrap** for statistical stability
- **Out-of-sample extension** (new data period)
- **Gate robustness**: vary gate threshold (0.0 → 0.1) and lookback (126/252/504)

---

### 7) How to Run

#### 7.1 Setup
Create an environment and install dependencies:
bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

#### 7.1 Setup 
python -m src.main
