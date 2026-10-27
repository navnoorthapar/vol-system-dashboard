"""
stats — statistical inference for the backtest and the forecasts.

    sharpe            PSR, DSR, MinTRL, Sharpe standard errors, HAC t-stats
    bootstrap         stationary-bootstrap intervals (Politis-Romano; PW block length)
    multiple_testing  Hansen SPA, Romano-Wolf StepM (arch), PBO by CSCV
    regression        OLS with Newey-West standard errors
    straddle_replay   bit-exact fast replay of the backtest's straddle P&L
    nulls             exposure-matched circular-shift / trade-shuffle timing nulls
    forecast          QLIKE/MSE, Diebold-Mariano, MCS, walk-forward vol baselines
    benchmarks        Cboe strategy indices and the PUT/WPUT implied β
    trials            the committed trials ledger (docs/trials_ledger.yaml)
    artifacts         deterministic JSON / parquet writing

Nothing here changes a backtest number: every module reads the engine's
outputs (or re-runs the engine's own pricing) and measures them.
"""
