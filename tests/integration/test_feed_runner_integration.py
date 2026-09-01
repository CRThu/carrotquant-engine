"""
集成测试：Feed 体系与 Engine 运行器深度集成 (test_feed_runner_integration.py)
"""

import pytest
import numpy as np
import polars as pl

from cq.engine import Engine, strategy, BarContext, ts_table, event_table, static_table


def test_feed_runner_multi_table_dict_access():
    """验证多表字典传入下策略通过 ctx.get('index')、ctx.get('valuation')、ctx.get('dragon_tiger') 的层级访问与防未来切片"""
    stock_df = pl.DataFrame({
        "timestamp": [1704067200000, 1704153600000, 1704240000000, 1704326400000],
        "symbol": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
        "open": [10.0, 10.5, 11.0, 11.5],
        "high": [11.0, 11.5, 12.0, 12.5],
        "low": [9.5, 10.0, 10.5, 11.0],
        "close": [10.2, 10.8, 11.2, 11.8],
    })

    index_df = pl.DataFrame({
        "timestamp": [1704067200000, 1704153600000, 1704240000000, 1704326400000],
        "symbol": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
        "open": [3000.0, 3050.0, 3100.0, 3150.0],
        "high": [3050.0, 3100.0, 3150.0, 3200.0],
        "low": [2980.0, 3020.0, 3080.0, 3120.0],
        "close": [3020.0, 3080.0, 3120.0, 3180.0],
    })

    val_df = pl.DataFrame({
        "timestamp": [1704067200000, 1704153600000, 1704240000000, 1704326400000],
        "symbol": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
        "pe_ttm": [12.0, 12.2, 12.5, 12.8],
        "pb": [1.1, 1.2, 1.3, 1.4],
    })

    dt_df = pl.DataFrame({
        "timestamp": [1704153600000],
        "symbol": ["000001.SZ"],
        "reason": ["机构买入"],
    })

    industry_df = pl.DataFrame({
        "symbol": ["000001.SZ", "000001.SZ"],
        "concept": ["信创", "金融科技"],
    })

    observed_index_prices = []
    observed_pe_values = []
    observed_dt_events = []

    @strategy
    def my_strategy(ctx: BarContext):
        # 1. 访问副 TS 行情表
        idx_table = ctx.get("index")
        observed_index_prices.append(idx_table.price)
        assert len(idx_table.close_history) == ctx.step + 1

        # 2. 访问时序因子表 (TableContext 属性与历史矩阵)
        val_table = ctx.get("valuation")
        pe_val = val_table.pe_ttm
        assert len(pe_val) == ctx.n_symbols
        observed_pe_values.append(float(pe_val[0]))
        assert val_table.pe_ttm_history.shape == (ctx.step + 1, ctx.n_symbols)
        assert val_table.get_history("pb").shape == (ctx.step + 1, ctx.n_symbols)

        # 3. 访问离散事件表
        dt_snap = ctx.get("dragon_tiger")
        if ctx.step == 1:
            assert bool(dt_snap) is True
            assert dt_snap["000001.SZ"]["reason"] == "机构买入"
            observed_dt_events.append(dt_snap["000001.SZ"]["reason"])
            ctx.buy(0, 1000)
        else:
            assert dt_snap["000001.SZ"] is None

        # 4. 访问静态属性表 (一对多概念列表)
        ind_container = ctx.get("industry")
        assert ind_container["000001.SZ"] == ["信创", "金融科技"]

    engine = Engine(initial_cash=100_000.0)
    result = engine.run(
        strategy=my_strategy,
        data={
            "stock": stock_df,
            "index": ts_table(index_df),
            "valuation": ts_table(val_df),
            "dragon_tiger": event_table(dt_df),
            "industry": static_table(industry_df),
        },
    )

    assert len(observed_index_prices) == 4
    assert observed_pe_values == [12.0, 12.2, 12.5, 12.8]
    assert observed_dt_events == ["机构买入"]
    assert result.trade_count == 1
    assert result.trade_logs["amount"][0] == 1000.0


def test_feed_runner_no_volume_pure_trading():
    """验证缺失 volume 与 amount 时的纯净撮合流程 (流动性限制自动跳过与价格有效可交易性)"""
    stock_df = pl.DataFrame({
        "timestamp": [1704067200000, 1704153600000],
        "symbol": ["000001.SZ", "000001.SZ"],
        "open": [10.0, 10.5],
        "high": [11.0, 11.5],
        "low": [9.5, 10.0],
        "close": [10.2, 10.8],
    })

    @strategy
    def no_vol_strat(ctx: BarContext):
        assert ctx.volume is None
        assert ctx.amount is None
        if ctx.step == 0:
            ctx.buy(0, 500)

    engine = Engine(initial_cash=50_000.0, max_volume_ratio=0.1)
    result = engine.run(strategy=no_vol_strat, data=stock_df)

    assert result.trade_count == 1
    assert result.trade_logs["amount"][0] == 500.0


def test_feed_runner_warmup_steps():
    """验证 warmup_steps 预热期过滤机制 (预热期更新指标与策略状态，但不产生撮合扣款)"""
    stock_df = pl.DataFrame({
        "timestamp": [1704067200000 + i * 86400000 for i in range(6)],
        "symbol": ["000001.SZ"] * 6,
        "open": [10.0 + i for i in range(6)],
        "high": [11.0 + i for i in range(6)],
        "low": [9.5 + i for i in range(6)],
        "close": [10.2 + i for i in range(6)],
    })

    steps_seen = []

    @strategy
    def warmup_strat(ctx: BarContext):
        steps_seen.append(ctx.step)
        ctx.buy(0, 100)

    engine = Engine(initial_cash=100_000.0)
    result = engine.run(
        strategy=warmup_strat,
        data=stock_df,
        warmup_steps=3,
    )

    assert steps_seen == [0, 1, 2, 3, 4, 5]
    assert result.trade_count == 3
    executed_steps = result.trade_logs["step_idx"].to_list()
    assert executed_steps == [3, 4, 5]
