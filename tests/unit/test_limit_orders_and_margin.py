"""
限价单 (Limit Order)、撤单 (Cancel Order) 与保证金率/融资融券利息单元测试
"""

import numpy as np
import pytest
from cq.engine import strategy, BarContext, Engine, MatchingMode, MarketData


@pytest.fixture
def sample_market_data():
    """构造 5 个时间步、2 只标的的基础测试数据"""
    timestamps = np.array(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    symbols = ["000001.SZ", "000002.SZ"]

    # (T=5, N=2)
    # Stock 0 价格动态: 10 -> 9 (Low 8.5) -> 11 (High 11.5) -> 12 -> 13
    # Stock 1 价格动态: 20 -> 21 -> 22 -> 23 -> 24
    open_price = np.array([
        [10.0, 20.0],
        [9.5, 20.5],
        [11.0, 21.5],
        [12.0, 22.5],
        [13.0, 23.5],
    ])
    high_price = np.array([
        [10.5, 20.5],
        [10.0, 21.0],
        [11.5, 22.0],
        [12.5, 23.0],
        [13.5, 24.0],
    ])
    low_price = np.array([
        [9.5, 19.5],
        [8.5, 20.0],
        [10.5, 21.0],
        [11.5, 22.0],
        [12.5, 23.0],
    ])
    close_price = np.array([
        [10.0, 20.0],
        [9.0, 20.5],
        [11.0, 21.5],
        [12.0, 22.5],
        [13.0, 23.5],
    ])

    return MarketData(
        timestamps=timestamps,
        symbols=symbols,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
    )


def test_buy_limit_order_trigger(sample_market_data):
    """测试买入限价单在到达目标价格时触发成交"""
    order_id_holder = {}

    @strategy
    def limit_strategy(ctx: BarContext):
        if ctx.step == 0:
            # 步 0 下买入限价单，价格为 8.8 元 (Step 0 的 Low 是 9.5，故 Step 0 不触发)
            oid = ctx.buy_limit(symbol_idx=0, amount=100, price=8.8)
            order_id_holder["oid"] = oid

    engine = Engine(initial_cash=100_000.0, fee_rate=0.0, min_fee=0.0, stamp_duty=0.0, slippage=0.0)
    res = engine.run(strategy=limit_strategy, data=sample_market_data)

    # Step 1 时 Low 达到 8.5 (<= 8.8)，限价单触发成交
    assert res.trade_count == 1
    trade_df = res.trade_logs
    assert trade_df["symbol"][0] == "000001.SZ"
    assert trade_df["amount"][0] == 100
    assert trade_df["price"][0] == 8.8
    assert trade_df["step_idx"][0] == 1  # 对应第 1 步触发


def test_sell_limit_order_trigger(sample_market_data):
    """测试卖无限价单在市场最高价达到或超越限价时触发成交"""
    @strategy
    def sell_limit_strat(ctx: BarContext):
        if ctx.step == 0:
            # Step 0 先买入 100 股标的 0 现货 (Step 0 Close 10.0)
            ctx.buy(symbol_idx=0, amount=100)
        elif ctx.step == 1:
            # Step 1 挂出卖无限价单，价格为 11.2 (Step 1 High 为 10.0，未能触发)
            ctx.sell_limit(symbol_idx=0, amount=100, price=11.2)

    engine = Engine(initial_cash=100_000.0, fee_rate=0.0, min_fee=0.0, stamp_duty=0.0, slippage=0.0)
    res = engine.run(strategy=sell_limit_strat, data=sample_market_data)

    # Step 2 的 High 达到 11.5 (>= 11.2)，卖无限价单成功在 Step 2 触发成交，成交价为 11.2
    assert res.trade_count == 2
    trade_df = res.trade_logs
    assert trade_df["side"][1] == "SELL"
    assert trade_df["price"][1] == 11.2
    assert trade_df["step_idx"][1] == 2


def test_cancel_order_multi_bar(sample_market_data):
    """测试跨 Bar 留存的限价单在后续 Bar 策略执行中被成功撤销"""
    order_id_holder = {}

    @strategy
    def multi_bar_cancel_strat(ctx: BarContext):
        if ctx.step == 0:
            # 挂一个非常低的限价单 (6.0)
            oid = ctx.buy_limit(symbol_idx=0, amount=100, price=6.0)
            order_id_holder["oid"] = oid
        elif ctx.step == 1:
            # 在 Step 1 决定撤单
            ctx.cancel_order(order_id_holder["oid"])

    engine = Engine(initial_cash=100_000.0, fee_rate=0.0, min_fee=0.0, stamp_duty=0.0, slippage=0.0)
    res = engine.run(strategy=multi_bar_cancel_strat, data=sample_market_data)

    # 订单已成功撤销，不产生任何成交
    assert res.trade_count == 0


def test_margin_interest_rate(sample_market_data):
    """测试做多融资年化利息扣减"""
    @strategy
    def margin_strategy(ctx: BarContext):
        if ctx.step == 0:
            # 买入超出可用资金的股票，产生 50,000 元现金透支
            ctx.buy(symbol_idx=0, amount=15_000)  # 15,000 * 10 = 150,000 元 (初始资金 100,000 -> cash = -50,000)

    # 设置做多保证金率 0.5 (允许 2 倍杠杆)，做多融资年化利率 6.3% (0.063)
    engine = Engine(
        initial_cash=100_000.0,
        fee_rate=0.0,
        min_fee=0.0,
        stamp_duty=0.0,
        slippage=0.0,
        long_margin_ratio=0.5,
        margin_interest_rate=0.063,
    )
    res = engine.run(strategy=margin_strategy, data=sample_market_data)

    # 验证交易成功触发
    assert res.trade_count == 1
    # 验证每日产生的融资利息已被从 cash_history 中扣除
    daily_r = 0.063 / 252.0
    expected_daily_interest = 50_000.0 * daily_r
    # 步 0 产生交易后，第 1 步 (次日跨日结算) cash 为 -50000 - expected_daily_interest
    assert res.cash_history[1] == pytest.approx(-50_000.0 - expected_daily_interest, rel=1e-5)


def test_borrow_interest_rate(sample_market_data):
    """测试做空融券年化利息扣除"""
    @strategy
    def short_strategy(ctx: BarContext):
        if ctx.step == 0:
            # 卖空 1000 股标的 0
            ctx.sell(symbol_idx=0, amount=1_000)  # 1000 * 10 = 10,000

    # 设置融券年化利率 8.4% (0.084)
    engine = Engine(
        initial_cash=100_000.0,
        fee_rate=0.0,
        min_fee=0.0,
        stamp_duty=0.0,
        slippage=0.0,
        short_margin_ratio=1.0,
        borrow_interest_rate=0.084,
    )
    res = engine.run(strategy=short_strategy, data=sample_market_data)

    assert res.trade_count == 1
    # 做空后 cash 为 110,000
    # 第 0 步做空持仓 1000 股，收盘价 10.0 -> 做空市值 10,000
    daily_borrow_r = 0.084 / 252.0
    expected_borrow_fee = 10_000.0 * daily_borrow_r
    # 第 1 步 (次日跨日结算) cash 扣除 1 日融券利息
    assert res.cash_history[1] == pytest.approx(110_000.0 - expected_borrow_fee, rel=1e-5)
