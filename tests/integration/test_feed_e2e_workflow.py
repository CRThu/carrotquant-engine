"""
端到端测试：全流程多表数据供给与回测执行 (test_feed_e2e_workflow.py)
"""

import pytest
import numpy as np
import polars as pl

from cq.engine import Engine, strategy, BarContext, ts_table, event_table, static_table


class MockDuckDataQuery:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def to_df(self) -> pl.DataFrame:
        return self._df


def test_feed_e2e_full_backtest_pipeline():
    """测试完整端到端回测流水线：多表 Duck 对象注入 -> 策略计算 -> 撮合交割 -> 绩效分析"""
    n_days = 30
    symbols = ["000001.SZ", "000002.SZ", "600000.SH"]
    base_ts = 1704067200000
    day_ms = 86400000

    timestamps = [base_ts + i * day_ms for i in range(n_days)]

    # 1. 构造多标的行情长表
    records = []
    np.random.seed(42)
    for sym_idx, sym in enumerate(symbols):
        base_p = 10.0 * (sym_idx + 1)
        for t_idx, ts in enumerate(timestamps):
            price_noise = np.sin(t_idx / 3.0) * 2.0
            close_val = max(1.0, base_p + price_noise + t_idx * 0.1)
            open_val = close_val - 0.2
            high_val = close_val + 0.5
            low_val = close_val - 0.5
            vol_val = 10000.0 + np.random.rand() * 5000.0

            records.append({
                "timestamp": ts,
                "symbol": sym,
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close_val,
                "volume": vol_val,
                "amount": vol_val * close_val,
                "back_adj_factor": 1.0 + t_idx * 0.01,
            })

    stock_df = pl.DataFrame(records)

    # 2. 构造指数基准表
    bench_records = []
    for t_idx, ts in enumerate(timestamps):
        b_close = 3000.0 + t_idx * 5.0
        bench_records.append({
            "timestamp": ts,
            "symbol": "000300.SH",
            "open": b_close - 10.0,
            "high": b_close + 20.0,
            "low": b_close - 20.0,
            "close": b_close,
        })
    bench_df = pl.DataFrame(bench_records)

    # 3. 构造龙虎榜事件表
    dt_records = [
        {"timestamp": timestamps[5], "symbol": "000001.SZ", "action": "BUY_SIGNAL"},
        {"timestamp": timestamps[15], "symbol": "000002.SZ", "action": "BUY_SIGNAL"},
    ]
    dt_df = pl.DataFrame(dt_records)

    # 4. 构造行业静态表
    ind_df = pl.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ", "600000.SH"],
        "industry": ["金融", "地产", "金融"],
    })

    # 5. 定义策略
    @strategy
    def multi_factor_strategy(ctx: BarContext):
        bench_table = ctx.get("benchmark")
        assert bench_table.price > 0
        dt_events = ctx.get("dragon_tiger")

        for i, sym in enumerate(ctx.positions):
            if dt_events and dt_events[i] and dt_events[i].get("action") == "BUY_SIGNAL":
                ctx.buy(symbol_idx=i, amount=1000)

            if ctx.positions[i] > 0 and ctx.step >= 25:
                ctx.sell(symbol_idx=i, amount=1000)

    # 6. 使用 Duck 对象和 LazyFrame 装载并回测
    engine = Engine(initial_cash=500_000.0, fee_rate=0.0003, slippage=0.0001)

    result = engine.run(
        strategy=multi_factor_strategy,
        data={
            "stock": MockDuckDataQuery(stock_df),   # Duck 对象 (.to_df)
            "benchmark": (bench_df.lazy(), "ts"),   # 显式声明为 TS 副行情表
            "dragon_tiger": event_table(dt_df),     # 显式声明为 稀疏事件表
            "industry": static_table(ind_df),       # 显式声明为 静态映射表
        },
    )

    # 7. 验证回测绩效指标与交易日志
    metrics = result.calc_metrics()
    assert metrics["initial_cash"] == 500_000.0
    assert metrics["total_trades"] == 4  # 2 次买入 + 2 次卖出
    assert metrics["total_fee"] > 0.0
    assert len(result.portfolio_df) == 30

    summary_text = result.summary()
    assert "回测绩效报告" in summary_text
    assert "总交易笔数" in summary_text
