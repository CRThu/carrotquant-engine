import numpy as np
import pytest
from cq.engine import Engine, MarketData, strategy, BarContext


def test_cross_chunk_limit_orders_persistence():
    d1 = MarketData(
        timestamps=np.array(['2020-12-30', '2020-12-31']),
        symbols=['000001.SZ'],
        open_price=np.array([[10.0], [10.0]]),
        high_price=np.array([[10.5], [10.5]]),
        low_price=np.array([[9.5], [9.5]]),
        close_price=np.array([[10.0], [10.0]]),
    )

    d2 = MarketData(
        timestamps=np.array(['2021-01-04', '2021-01-05']),
        symbols=['000001.SZ'],
        open_price=np.array([[10.0], [8.0]]),
        high_price=np.array([[10.0], [8.5]]),
        low_price=np.array([[9.0], [7.5]]),
        close_price=np.array([[9.5], [8.0]]),
    )

    @strategy
    def limit_strat(ctx: BarContext):
        if ctx.datetime.startswith('2020-12-30'):
            ctx.buy_limit('000001.SZ', 100, price=8.5)

    engine = Engine(initial_cash=100_000.0, fee_rate=0.0, min_fee=0.0, stamp_duty=0.0, slippage=0.0)
    res = engine.run(strategy=limit_strat, data=[d1, d2])

    assert res.trade_count == 1
    assert res.trade_logs['price'][0] == 8.0
    assert res.trade_logs['datetime'][0].startswith('2021-01-05')
