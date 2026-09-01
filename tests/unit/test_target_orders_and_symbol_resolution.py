import numpy as np
import pytest
from cq.engine import Engine, MarketData, strategy, BarContext


def test_target_orders_and_t1_constraints():
    timestamps = np.array([
        1704067200000,
        1704067200000 + 60000,
        1704067200000 + 86400000,
        1704067200000 + 86400000 + 60000
    ], dtype=np.int64)
    symbols = ['000001.SZ', '600000.SH']
    close_p = np.array([
        [10.0, 20.0],
        [10.0, 20.0],
        [10.0, 20.0],
        [10.0, 20.0]
    ])

    data = MarketData(
        timestamps=timestamps,
        symbols=symbols,
        open_price=close_p,
        high_price=close_p,
        low_price=close_p,
        close_price=close_p,
    )

    @strategy
    def target_strat(ctx: BarContext):
        if ctx.step == 0:
            ctx.order_target_amount('000001.SZ', 200)
            ctx.order_target_value('600000.SH', 2000.0)
        elif ctx.step == 1:
            ctx.sell('000001.SZ', 100)
        elif ctx.step == 2:
            ctx.sell('000001.SZ', 100)

    engine = Engine(
        initial_cash=100_000.0,
        fee_rate=0.0,
        min_fee=0.0,
        stamp_duty=0.0,
        slippage=0.0,
        enable_t1=True,
    )
    res = engine.run(strategy=target_strat, data=data)

    assert res.trade_count == 3
    assert res.trade_logs['symbol'][0] == '000001.SZ'
    assert res.trade_logs['amount'][0] == 200
    assert res.trade_logs['symbol'][1] == '600000.SH'
    assert res.trade_logs['amount'][1] == 100
    assert res.trade_logs['symbol'][2] == '000001.SZ'
    assert res.trade_logs['step_idx'][2] == 2
