import numpy as np
import pytest
from cq.engine import Engine, MarketData, strategy, BarContext


def test_multi_chunk_ipo_universe_expansion():
    d1 = MarketData(
        timestamps=np.array(['2020-01-01', '2020-01-02']),
        symbols=['000001.SZ', '000002.SZ'],
        open_price=np.array([[10.0, 20.0], [10.5, 20.5]]),
        high_price=np.array([[10.5, 20.5], [11.0, 21.0]]),
        low_price=np.array([[9.5, 19.5], [10.0, 20.0]]),
        close_price=np.array([[10.0, 20.0], [10.5, 20.5]]),
    )

    d2 = MarketData(
        timestamps=np.array(['2021-01-01', '2021-01-02']),
        symbols=['000001.SZ', '000002.SZ', '000003.SZ'],
        open_price=np.array([[11.0, 21.0, 30.0], [11.5, 21.5, 30.5]]),
        high_price=np.array([[11.5, 21.5, 30.5], [12.0, 22.0, 31.0]]),
        low_price=np.array([[10.5, 20.5, 29.5], [11.0, 21.0, 30.0]]),
        close_price=np.array([[11.0, 21.0, 30.0], [11.5, 21.5, 30.5]]),
    )

    @strategy
    def my_strat(ctx: BarContext):
        if ctx.datetime.startswith('2020-01-01') and ctx.get_position('000001.SZ') == 0:
            ctx.buy('000001.SZ', 100)
        elif ctx.datetime.startswith('2021-01-01'):
            assert ctx.get_position('000001.SZ') == 100
            assert ctx.get_position('000003.SZ') == 0
            ctx.sell('000001.SZ', 100)
            ctx.buy('000003.SZ', 200)

    engine = Engine(initial_cash=100_000.0, fee_rate=0.0, min_fee=0.0, stamp_duty=0.0, slippage=0.0)
    res = engine.run(strategy=my_strat, data=[d1, d2])

    assert res.trade_count == 3
    assert set(res.symbols) == {'000001.SZ', '000002.SZ', '000003.SZ'}
    assert res.portfolio_value[-1] > 100_000.0


def test_multi_chunk_delisting_and_reordering():
    d1 = MarketData(
        timestamps=np.array(['2020-01-01', '2020-01-02']),
        symbols=['AAA', 'BBB', 'CCC'],
        open_price=np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]),
        high_price=np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]),
        low_price=np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]),
        close_price=np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]),
    )

    d2 = MarketData(
        timestamps=np.array(['2021-01-01', '2021-01-02']),
        symbols=['CCC', 'AAA'],
        open_price=np.array([[30.0, 10.0], [30.0, 10.0]]),
        high_price=np.array([[30.0, 10.0], [30.0, 10.0]]),
        low_price=np.array([[30.0, 10.0], [30.0, 10.0]]),
        close_price=np.array([[30.0, 10.0], [30.0, 10.0]]),
    )

    @strategy
    def my_strat(ctx: BarContext):
        if ctx.datetime.startswith('2020-01-01'):
            ctx.buy('AAA', 100)
            ctx.buy('CCC', 500)
        elif ctx.datetime.startswith('2021-01-01'):
            assert ctx.get_position('AAA') == 100
            assert ctx.get_position('CCC') == 500

    engine = Engine(initial_cash=100_000.0, fee_rate=0.0, min_fee=0.0, stamp_duty=0.0, slippage=0.0)
    res = engine.run(strategy=my_strat, data=[d1, d2])
    assert res.trade_count == 2
