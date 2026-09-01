"""
单元测试：MarketData 与 AdjMarketData 复权视图行为测试
"""

import numpy as np
import pytest
from cq.engine.feed.column_loader import MarketData


def test_market_data_unadjusted_and_adjusted_views():
    timestamps = np.array(["2024-01-01", "2024-01-02"])
    symbols = ["AAPL", "GOOG"]

    # 原始未复权价
    raw_close = np.array([[100.0, 200.0], [105.0, 195.0]])
    open_p = raw_close * 0.99
    high_p = raw_close * 1.01
    low_p = raw_close * 0.98

    # 后复权价 (复权因子 = 2.0)
    adj_close = raw_close * 2.0
    adj_open = open_p * 2.0
    adj_high = high_p * 2.0
    adj_low = low_p * 2.0

    data = MarketData(
        timestamps=timestamps,
        symbols=symbols,
        open_price=open_p,
        high_price=high_p,
        low_price=low_p,
        close_price=raw_close,
        adj_close_price=adj_close,
        adj_open_price=adj_open,
        adj_high_price=adj_high,
        adj_low_price=adj_low,
    )

    assert data.n_symbols == 2
    assert data.n_steps == 2

    # 真实交易价格
    np.testing.assert_array_equal(data.close, raw_close)

    # 复权视图价格
    np.testing.assert_array_equal(data.adj.close, adj_close)
    np.testing.assert_array_equal(data.adj.open, adj_open)
