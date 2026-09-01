"""
集成测试：端到端完整回测工作流测试 (Backtest Workflow)
"""

import pytest
import numpy as np
from cq.engine import strategy, BarContext, Engine
from cq.engine.feed.column_loader import MarketDataContainer


def test_full_backtest_workflow():
    # 构造 100 个时间步，5 只股票的模拟行情矩阵
    np.random.seed(42)
    n_steps = 100
    n_stocks = 5
    timestamps = np.array([f"2024-01-{i+1:02d}" for i in range(n_steps)])
    symbols = [f"00000{i}.SZ" for i in range(1, n_stocks + 1)]

    base_prices = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    price_changes = np.random.randn(n_steps, n_stocks) * 0.5
    close_p = base_prices + np.cumsum(price_changes, axis=0)
    close_p = np.maximum(close_p, 1.0)
    open_p = close_p * 0.99
    high_p = close_p * 1.01
    low_p = close_p * 0.98
    vol = np.full((n_steps, n_stocks), 10000.0)
    amt = close_p * vol

    data = MarketDataContainer(
        timestamps=timestamps,
        symbols=symbols,
        open_price=open_p,
        high_price=high_p,
        low_price=low_p,
        close_price=close_p,
        volume=vol,
        amount=amt,
    )

    # 定义简单均值回归策略：逢低买入，逢高卖出
    @strategy
    def mean_reversion_strategy(ctx: BarContext):
        if ctx.step < 5:
            return

        for i in range(ctx.n_symbols):
            if not ctx.is_tradable[i]:
                continue
            history_close = ctx.close_history[-5:, i]
            ma5 = np.mean(history_close)
            curr_p = ctx.close[i]

            # 价格低于 5日均线 并且未持仓，则买入 100 股
            if curr_p < ma5 and ctx.positions[i] == 0:
                ctx.buy(symbol_idx=i, amount=100)
            # 价格高于 5日均线 并且有持仓，则卖出
            elif curr_p > ma5 and ctx.positions[i] > 0:
                ctx.sell(symbol_idx=i, amount=ctx.positions[i])

    engine = Engine(initial_cash=100_000.0, fee_rate=0.0003, min_fee=5.0)
    results = engine.run(strategy=mean_reversion_strategy, data=data)

    assert results.initial_cash == 100_000.0
    assert results.portfolio_value.shape == (n_steps,)
    assert results.trade_count > 0

    summary_text = results.summary()
    assert "回测绩效报告" in summary_text
    assert "累计收益率" in summary_text

    # 校验交易日志导出 Polars DataFrame
    logs_df = results.trade_logs
    assert len(logs_df) == results.trade_count
    assert "symbol" in logs_df.columns
    assert "cash_after" in logs_df.columns
