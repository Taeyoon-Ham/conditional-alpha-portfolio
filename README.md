# Conditional Alpha Portfolio (SPY Core + Gated Overlay)


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

#### 7.2 Execute 
python -m src.main

---

## 한국어

### 1) 개요 (Executive Summary)
이 저장소는 **연구(리서치) 수준의 백테스트**로,  
**SPY를 코어(Core)로 유지하면서 조건부 알파(Conditional Alpha)를 오버레이(Overlay) 형태로 추가**하는 포트폴리오 전략을 구현합니다.

핵심 특징은 다음과 같습니다.

- **코어(Core)**: SPY 100% (항상 시장 베타 유지)
- **오버레이(Overlay)**: QQQ / IWM / TLT / GLD로 구성된 소규모 전술 포지션
- **Crowding 필터**: 시장이 덜 혼잡한(low crowding) 구간에서만 오버레이 허용
- **게이트(Gate)**: 최근 1년(252영업일) 동안 오버레이의 액티브 IR이 양(+)일 때만 오버레이 활성화
- **주간 리밸런스(5영업일)**: 과도한 매매와 거래비용을 억제
- **탐색 방식**: 그리드 탐색 없이, 상한이 있는 랜덤 서치 + 조기 종료

목표는 **시장 노출을 포기하지 않으면서**,  
알파가 작동하는 조건에서만 **작고 통제된 방식으로 초과수익을 추구**하는 것입니다.

---

### 2) 전략 정의 (Strategy Definition)

#### 2.1 투자 유니버스
본 전략은 다음 ETF들로 구성됩니다.

- **SPY**: 미국 대형주 지수 (벤치마크이자 코어 자산)
- **QQQ**: 성장주/테크 편향
- **IWM**: 미국 중소형주
- **TLT**: 미국 장기국채 (리스크 오프 구간 방어)
- **GLD**: 금 (위기 국면 분산 자산)

모든 자산은 유동성이 충분한 ETF로 가정합니다.

---

#### 2.2 코어(Core) + 오버레이(Overlay) 구조
기본 아이디어는 다음과 같습니다.

- **항상 SPY를 100% 보유**하여 장기 시장 성장에 참여
- 추가 수익은 **오버레이**라는 작은 전술 포지션으로만 추구
- 오버레이는 조건이 맞을 때만 켜지고, 그렇지 않으면 자동으로 꺼짐

수식으로 표현하면:

- \( r_{t,i} \): t일의 i자산 일간 수익률  
- \( \mathbf{w}^{core}_t \): 코어 비중 (SPY = 1.0)  
- \( \mathbf{w}^{ov}_t \): 오버레이 비중  
- \( k_t \in \{0, k\} \): 게이트에 따른 오버레이 스케일  

전체 포트폴리오 비중:
\[
\mathbf{w}_t = \mathbf{w}^{core}_t + k_t \cdot \mathbf{w}^{ov}_t
\]

일간 포트폴리오 수익률:
\[
r^{port}_t = \sum_i w_{t,i} \cdot r_{t,i} - \text{거래비용}_t
\]

---

### 3) 시장 국면 판단 (Regime Logic)

#### 3.1 추세(Trend) 필터
SPY 가격을 기준으로 두 개의 이동평균을 계산합니다.

- 빠른 이동평균: `ma_fast`
- 느린 이동평균: `ma_slow`

판단 기준:
- `ma_fast > ma_slow` → 상승 국면(Bull)
- 그렇지 않으면 → 하락 국면(Bear)

#### 3.2 모멘텀(Momentum) 필터
SPY의 중기 모멘텀을 다음과 같이 정의합니다.

\[
mom(t) = \frac{P_{SPY}(t)}{P_{SPY}(t - L)} - 1
\]

- 기본값: `L = 126` (약 6개월)
- 상승 국면 + 모멘텀 양(+) → **BullStrong**

---

### 4) 오버레이 포지션 규칙

#### 4.1 상승 국면 (BullStrong + Low Crowding)
- QQQ / IWM 비중으로 공격적 오버레이 구성
- QQQ 비중 = `qqq_w`
- IWM 비중 = `1 - qqq_w`

#### 4.2 하락 국면 (Bear + Low Crowding)
- 방어적 자산으로 이동
- 단, TLT가 자기 이동평균 위에 있을 때만 허용

조건:
\[
P_{TLT}(t) > MA_{126}(TLT)
\]

비중:
- TLT = `bear_tlt_w`
- GLD = `bear_gld_w`

#### 4.3 그 외의 경우
- 오버레이 = 0  
(즉, SPY만 보유)

---

### 5) Crowding 필터 (혼잡도)
Crowding Score는 **시장 참여자들이 한 방향으로 얼마나 몰려 있는지**를 나타내는 지표입니다.

- crowding score가 하위 \( q \) 분위수 이하일 때만 오버레이 허용
- 파라미터: `crowd_q` (예: 0.80, 0.85, 0.90)

\[
LowCrowd(t) \iff Crowd(t) \leq Q_q(Crowd)
\]

이는 “모두가 같은 포지션을 잡은 위험한 구간”을 피하기 위한 장치입니다.

---

### 6) 게이트(Gate): 액티브 IR 기반 필터
오버레이는 언제나 유효하지 않습니다.  
따라서 최근 성과가 나쁜 오버레이는 **자동으로 비활성화**합니다.

- 오버레이 수익을 액티브 수익으로 간주
- 최근 252일 기준 IR 계산

\[
IR_t = \frac{\mu_t}{\sigma_t} \sqrt{252}
\]

규칙:
- \( IR_t > 0 \) → 오버레이 ON
- \( IR_t \le 0 \) → 오버레이 OFF

이 구조는 전략을 **적응형(adaptive)**으로 만듭니다.

---

### 7) 리밸런스 및 리스크 관리

#### 7.1 주간 리밸런스
- 매일 계산은 하되, **실제 비중 변경은 5영업일마다**
- 거래비용과 턴오버를 실질적으로 감소

#### 7.2 익스포저 제한
- 오버레이 총 비중 한도
\[
\sum_i |w^{ov}_{t,i}| \le \text{overlay\_gross\_cap}
\]

- 전체 포트폴리오 레버리지 한도
\[
\sum_i |w_{t,i}| \le \text{lev\_cap}
\]

---

### 8) 백테스트 설계 (Experiment Design)

#### 8.1 Train / Test 분리
- **Selection (Train)**: 파라미터 선택용
- **Holdout (Test)**: 성과 검증용

Test 구간은 절대 파라미터 탐색에 사용하지 않습니다.

#### 8.2 탐색 방식
- 랜덤 서치(Random Search)
- 후보 수 상한 (`N_CANDIDATES`)
- 개선 없으면 조기 종료 (`EARLY_STOP_PATIENCE`)

과도한 최적화를 방지하기 위한 현실적인 접근입니다.

---

### 9) 성과 지표 해석

- **Total Return**: 누적 수익
- **CAGR**: 연복리 수익률
- **Volatility**: 연환산 변동성
- **Sharpe Ratio**: 위험 대비 수익
- **Max Drawdown(MDD)**: 최대 낙폭
- **Alpha / Beta**: SPY 대비 회귀 분석 결과
- **Information Ratio(IR)**: 액티브 성과 품질
- **Turnover**: 매매 빈도 (실전성 판단 핵심)

또한 혼잡도 구간별(LOW / HIGH) 성과를 분리 분석하여  
**조건부 알파가 실제로 특정 구간에서만 발생하는지**를 검증합니다.

---

### 10) 한계와 추가 검증 과제

이 코드는 연구 목적이며, 다음 한계가 존재합니다.

- 슬리피지 단순화
- ETF 유동성 가정
- 과최적화 가능성

권장 추가 검증:
- Walk-forward 분석
- 거래비용 스트레스 테스트
- 파라미터 안정성 분석
- 신규 데이터 구간 OOS 테스트

---

### 11) 실행 방법

```bash
python -m src.main

