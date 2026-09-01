"""
单元测试：@strategy 装饰器、BarContext 与物理防未来切片
"""

import pytest
import numpy as np
from cq.engine.strategy.base import strategy
from cq.engine.strategy.context import BarContext


def test_strategy_decorator():
    called = False

    @strategy
    def mock_strat(ctx: BarContext):
        nonlocal called
        called = True
        ctx.buy(0, 100)

    assert getattr(mock_strat, "_is_carrot_strategy", False) is True


def test_bar_context_bound_checking():
    timestamps = np.array(["2024-01-01", "2024-01-02", "2024-01-03"])
    open_p = np.array([[10.0], [11.0], [12.0]])
    close_p = np.array([[10.5], [11.5], [12.5]])
    is_t = np.array([[True], [True], [True]])
    pos = np.zeros(1)

    orders = []
    # 模拟在时间步 t=1 (对应第2个元素)
    ctx = BarContext(
        step=1,
        n_symbols=1,
        timestamps=timestamps,
        open_mat=open_p,
        high_mat=open_p,
        low_mat=open_p,
        close_mat=close_p,
        volume_mat=open_p,
        amount_mat=open_p,
        is_tradable_mat=is_t,
        positions=pos,
        cash=10000.0,
        orders_buffer=orders,
    )

    # 1. 正常读取历史切片 [:2]
    assert len(ctx.close_history) == 2
    assert ctx.close_history[1, 0] == 11.5

    # 2. 读取当前 Bar 单标的价格
    assert ctx.price == 11.5

    # 3. 试图在物理内存切片上访问 t+1 (即索引 2) 必然触发 IndexError
    with pytest.raises(IndexError):
        _ = ctx.close_history[2, 0]

    # 4. 尝试挂单
    ctx.buy_single(100)
    assert len(orders) == 1
    assert orders[0] == (1, 0, 100.0)

