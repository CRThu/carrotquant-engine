import numpy as np
import pytest
from cq.engine import Engine, MarketData, strategy, BarContext


def test_metrics_negative_equity_protection():
    timestamps = np.array(['2024-01-01', '2024-01-02', '2024-01-03'])
    symbols = ['000001.SZ']
    open_p = np.array([[10.0], [5.0], [1.0]])
    high_p = np.array([[10.0], [5.0], [1.0]])
    low_p = np.array([[10.0], [5.0], [1.0]])
    close_p = np.array([[10.0], [5.0], [1.0]])

    data = MarketData(
        timestamps=timestamps,
        symbols=symbols,
        open_price=open_p,
        high_price=high_p,
        low_price=low_p,
        close_price=close_p,
    )

    @strategy
    def bankruptcy_strat(ctx: BarContext):
        if ctx.step == 0:
            ctx.buy(0, 50000)

    engine = Engine(initial_cash=100_000.0, long_margin_ratio=0.1, fee_rate=0.0, min_fee=0.0, stamp_duty=0.0, slippage=0.0)
    res = engine.run(strategy=bankruptcy_strat, data=data)

    m = res.calc_metrics()
    assert m['annualized_return'] == -1.0
    assert not np.isnan(m['max_drawdown'])
    assert m['max_drawdown'] > 1.0 or m['max_drawdown'] == pytest.approx(1.0, rel=0.1)
    summary_text = res.summary()
    assert '胜率' in summary_text
    assert '卡玛比率' in summary_text
