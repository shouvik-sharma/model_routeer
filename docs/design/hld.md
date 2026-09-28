# QuantMetron High Level Design (HLD)

## 1. Goal / Non-goals
**Goal:** AI tutor that onboards non-retail/aspiring retail traders to systematic thinking via daily lesson + same-day simulation + 1y backtest + 1-week paper trial, no real money.

**Non-goals for MVP:**
- No live brokerage, no real orders, no personalized financial advice.
- No intraday tick data — daily OHLC only.
- No custom backtest engine — use OSS `backtesting.py`.

**Hackathon Fit:** Best Apps and Agents Track.
- LLM: Nemotron (Ultra for reasoning, Nano/Super for chat) on Nebius Token Factory.
- Execution: Backtests run in Token Factory Sandboxes.
- Hosting: API on Nebius Serverless Endpoints.

> Safety banner: Educational project only. Not financial advice. Simulated results ≠ future returns.

## 2. Human -> Agent Flows

### A. Onboarding Decision Tree
- **Start** -> Are you first-time on Quant Path? [Yes/No]
  - **No** -> Load profile from memory -> Resume at Day N -> Daily Loop
  - **Yes** -> Choose path: [1-week | 1-month | 3-month]
       -> Self-rate quant: [Beginner | Advanced | Pro]
       -> Self-rate options: [Beginner | Advanced | Pro]
       -> Agent: "Perfect, this program is education-only..." + show syllabus preview + Day 1 topic
       -> Create profile + start Daily Loop

### B. Everyday Loop
1. **Recall**: Load last N turns + profile + yesterday's feedback/paper P&L.
2. **Teach**: Pick next topic from syllabus (no repeats, adapt to level).
3. **Explain**: General terms + numeric example.
4. **Simulate today**: Run strategy on today's daily bar.
5. **Backtest 1y**: Same strategy on last 1y daily data via `backtesting.py`.
6. **Assign 1-week paper trial**: e.g. SMA(20/50) on SPY. User paper-trades via CLI.
7. **Collect feedback**: Confidence 1-5 + notes -> save to memory.

## 3. Curriculum Matrix (Equities + Options)
Syllabus ordered DAG filtered by `path_len x level`.

**1-week (MVP Demo):**
1. Risk, expectancy, position sizing
2. Trend: SMA cross (equity backtest)
3. Mean-reversion: RSI(14)
4. Volatility + drawdown reading
5. Options 101: call/put payoff + covered call
6. Cash-secured put + vertical spread
7. Journal/discipline + interpreting backtests

## 4. Architecture (API + CLI)
`[Typer CLI] <-> [FastAPI API] <-> [Agent Orchestrator - LangGraph]`
- **LLM Router**: Nebius Token Factory / Nemotron Ultra + Nano.
- **Memory**: SQLite + vector recall (history, profile, feedback).
- **Curriculum Store**: YAML syllabus.
- **Tools**:
  - `market-data.py`: Stooq/Yahoo daily cache.
  - `equity-backtest.py`: `backtesting.py` in Sandbox.
  - `options-payoff.py`: Black-Scholes + historical underlying.
  - `paper-portfolio.py`: Virtual ledger.

## 5. Data Contracts
- `POST /onboard {first_time, path_len, quant_level, options_level}`
- `POST /daily {user_id}` -> `{topic, explanation, today_sim, backtest_1y{metrics, chart_url}, trial_assignment}`
- `POST /feedback {user_id, day, confidence 1-5, note, paper_pnl}`

## 6. LLM / Guardrails
- Socratic tutor, no live trade recommendations, mandatory risk disclaimers.
- Ultra for planning/explaining; Nano for chat/summaries.
- Restricted tool-calling: No shell access, no real-money orders.

## 7. MVP Build Plan
1. API + CLI skeleton + SQLite profile.
2. Curriculum YAML for 1-week path.
3. `market.py` + `backtesting.py` wrappers.
4. `options.py` payoff calculator.
5. Agent graph (Memory -> Teach -> Tool -> Assign -> Feedback).
6. Nebius deployment + Demo recording.

## 8. Risks
- Yahoo/Stooq rate limits -> Use CSV cache.
- LLM hallucinations -> Render tool JSON verbatim.
- Options complexity -> MVP focuses on payoff at expiry.
