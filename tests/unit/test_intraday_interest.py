import numpy as np
import pytest
from cq.engine import Engine, MarketData, strategy, BarContext


def test_intraday_minute_bars_interest_rate():
    n_steps = 240
    timestamps = np.array([1704067200000 + i * 60000 for i in range(n_steps)], dtype=np.int64)
    symbols = ['000001.SZ']
    close_p = np.full((n_steps, 1), 10.0)

    data = MarketData(
        timestamps=timestamps,
        symbols=symbols,
        open_price=close_p,
        high_price=close_p,
        low_price=close_p,
        close_price=close_p,
    )

    @strategy
    def margin_strat(ctx: BarContext):
        if ctx.step == 0:
            ctx.buy(0, 15000)

    engine = Engine(
        initial_cash=100_000.0,
        fee_rate=0.0,
        min_fee=0.0,
        stamp_duty=0.0,
        slippage=0.0,
        long_margin_ratio=0.5,
        margin_interest_rate=0.063,
    )
    res = engine.run(strategy=margin_strat, data=data)

    assert res.trade_count == 1
    assert res.cash_history[0] == -50000.0
    assert res.cash_history[-1] == -50000.0
