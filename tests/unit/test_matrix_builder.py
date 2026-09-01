"""
单元测试：单趟极速矩阵构建器与离散/静态容器全方位测试 (test_matrix_builder.py)
"""

from datetime import date, datetime
import pytest
import numpy as np
import polars as pl

from cq.engine.feed.matrix_builder import (
    build_market_data_from_df,
    build_event_container_from_df,
    build_static_container_from_df,
    parse_timestamps_series,
    EventSnapshot,
    SparseEventContainer,
    StaticAttributeContainer,
    AdjMarketData,
    LazyCustomFields,
    CustomFieldsDict,
    MarketData,
)


def test_matrix_builder_basic_and_c_contiguous():
    """测试单趟矩阵填充与内存连续性 (C-Contiguous 标志验证)"""
    df = pl.DataFrame({
        "timestamp": [1704067200000, 1704153600000, 1704067200000, 1704153600000],
        "symbol": ["000001.SZ", "000001.SZ", "600000.SH", "600000.SH"],
        "open": [10.0, 10.5, 20.0, 20.5],
        "high": [11.0, 11.5, 21.0, 21.5],
        "low": [9.5, 10.0, 19.5, 20.0],
        "close": [10.2, 10.8, 20.2, 20.8],
        "volume": [1000.0, 1200.0, 2000.0, 2200.0],
        "amount": [10200.0, 12960.0, 40400.0, 45760.0],
        "back_adj_factor": [1.0, 1.05, 1.0, 1.05],
        "pe_ttm": [12.5, 12.8, 8.5, 8.6],
    })

    data = build_market_data_from_df(df, columns=["pe_ttm"])

    assert data.shape == (2, 2)
    assert data.n_steps == 2
    assert data.n_symbols == 2
    assert data.symbols == ["000001.SZ", "600000.SH"]
    assert "MarketData" in repr(data)
    assert data["pe_ttm"] is not None

    # C-Contiguous 内存连续性验证
    assert data.open.flags.c_contiguous
    assert data.high.flags.c_contiguous
    assert data.low.flags.c_contiguous
    assert data.close.flags.c_contiguous
    assert data.volume.flags.c_contiguous
    assert data.amount.flags.c_contiguous
    assert data.adj_factor.flags.c_contiguous

    # 数据准确性断言
    np.testing.assert_allclose(data.open, [[10.0, 20.0], [10.5, 20.5]])
    np.testing.assert_allclose(data.close, [[10.2, 20.2], [10.8, 20.8]])
    np.testing.assert_allclose(data.volume, [[1000.0, 2000.0], [1200.0, 2200.0]])

    # 自定义字段单趟填充验证
    assert "pe_ttm" in data.custom_fields
    assert data.custom_fields["pe_ttm"].flags.c_contiguous
    np.testing.assert_allclose(data.custom_fields["pe_ttm"], [[12.5, 8.5], [12.8, 8.6]])

    # 验证未找到列时抛出 KeyError
    with pytest.raises(KeyError, match="未找到自定义列"):
        data.get_custom_field("non_existent")


def test_matrix_builder_mandatory_ohlc_validation():
    """测试 OHLC 强制性校验：缺失任意一列或时间/标的列时必须抛出 ValueError"""
    valid_base = {
        "timestamp": [1704067200000],
        "symbol": ["000001.SZ"],
        "open": [10.0],
        "high": [11.0],
        "low": [9.5],
        "close": [10.2],
    }

    # 1. 缺失 symbol
    no_symbol = pl.DataFrame({k: v for k, v in valid_base.items() if k != "symbol"})
    with pytest.raises(ValueError, match="缺失必需的标的代码列"):
        build_market_data_from_df(no_symbol)

    # 2. 缺失时间列
    no_time = pl.DataFrame({k: v for k, v in valid_base.items() if k != "timestamp"})
    with pytest.raises(ValueError, match="缺失必需的时间列"):
        build_market_data_from_df(no_time)

    # 3. 缺失 open
    no_open = pl.DataFrame({k: v for k, v in valid_base.items() if k != "open"})
    with pytest.raises(ValueError, match="缺失必需的 OHLC 价格列"):
        build_market_data_from_df(no_open)

    # 4. 缺失 high
    no_high = pl.DataFrame({k: v for k, v in valid_base.items() if k != "high"})
    with pytest.raises(ValueError, match="缺失必需的 OHLC 价格列"):
        build_market_data_from_df(no_high)

    # 5. 缺失 low
    no_low = pl.DataFrame({k: v for k, v in valid_base.items() if k != "low"})
    with pytest.raises(ValueError, match="缺失必需的 OHLC 价格列"):
        build_market_data_from_df(no_low)

    # 6. 缺失 close
    no_close = pl.DataFrame({k: v for k, v in valid_base.items() if k != "close"})
    with pytest.raises(ValueError, match="缺失必需的 OHLC 价格列"):
        build_market_data_from_df(no_close)

    # 7. 空 DataFrame
    with pytest.raises(ValueError, match="无法从空 DataFrame 构建"):
        build_market_data_from_df(pl.DataFrame())


def test_matrix_builder_none_volume_and_amount():
    """测试 volume 与 amount 未提供时的纯净 None 状态表示"""
    df = pl.DataFrame({
        "timestamp": [1704067200000, 1704153600000],
        "symbol": ["000001.SZ", "000001.SZ"],
        "open": [10.0, 10.5],
        "high": [11.0, 11.5],
        "low": [9.5, 10.0],
        "close": [10.2, 10.8],
    })

    data = build_market_data_from_df(df)
    assert data.volume is None
    assert data.amount is None
    assert np.all(data.is_tradable)


def test_matrix_builder_sparse_bars_nan_padding():
    """测试非连续稀疏交易日下自动填充 NaN 保持坐标系对齐"""
    # 000001.SZ 有两个交易日，600000.SH 仅在第二个交易日有数据
    df = pl.DataFrame({
        "timestamp": [1704067200000, 1704153600000, 1704153600000],
        "symbol": ["000001.SZ", "000001.SZ", "600000.SH"],
        "open": [10.0, 10.5, 20.0],
        "high": [11.0, 11.5, 21.0],
        "low": [9.5, 10.0, 19.5],
        "close": [10.2, 10.8, 20.2],
    })

    data = build_market_data_from_df(df)
    assert data.shape == (2, 2)
    # 600000.SH 在第 0 步停牌/无数据，应为 NaN
    assert np.isnan(data.open[0, 1])
    assert np.isnan(data.close[0, 1])
    assert not data.is_tradable[0, 1]
    # 在第 1 步有数据
    assert data.close[1, 1] == 20.2
    assert data.is_tradable[1, 1]


def test_parse_timestamps_series_datetime_strings():
    """测试 parse_timestamps_series 对多种 datetime 格式的解析"""
    # 1. 毫秒时间戳
    df_ts = pl.DataFrame({"timestamp": [1704067200000, 1704153600000]})
    res_ts = parse_timestamps_series(df_ts)
    assert res_ts[0] == 1704067200000

    # 2. ISO 格式 Datetime
    df_iso = pl.DataFrame({"datetime": ["2024-01-01 09:30:00", "2024-01-02 09:30:00"]})
    res_iso = parse_timestamps_series(df_iso)
    assert res_iso.dtype == pl.Int64
    assert len(res_iso) == 2

    # 3. Python Date / Datetime 格式
    df_date_type = pl.DataFrame({"datetime": [date(2024, 1, 1), date(2024, 1, 2)]})
    res_dt = parse_timestamps_series(df_date_type)
    assert res_dt.dtype == pl.Int64
    assert len(res_dt) == 2

    # 4. YYYY-MM-DD 格式
    df_date = pl.DataFrame({"datetime": ["2024-01-01", "2024-01-02"]})
    res_date = parse_timestamps_series(df_date)
    assert res_date.dtype == pl.Int64
    assert len(res_date) == 2


def test_sparse_event_container_comprehensive():
    """测试 SparseEventContainer 与 EventSnapshot 离散事件的稀疏索引、字典接口与 None 返回"""
    df = pl.DataFrame({
        "timestamp": [1704067200000, 1704067200000, 1704153600000],
        "symbol": ["000001.SZ", "600000.SH", "600000.SH"],
        "reason": ["龙虎榜买入", "大宗交易溢价", "机构买入"],
        "net_amount": [5000000.0, 20000000.0, 12000000.0],
    })

    timestamps = np.array([1704067200000, 1704153600000, 1704240000000], dtype=np.int64)
    symbols = ["000001.SZ", "600000.SH", "000002.SZ"]

    container = build_event_container_from_df(df, timestamps, symbols)
    assert "SparseEventContainer" in repr(container)

    # Step 0: 000001.SZ 与 600000.SH 均发生事件，000002.SZ 无事件
    snap0 = container.get_snapshot(0)
    assert "EventSnapshot" in repr(snap0)
    assert bool(snap0) is True
    assert len(snap0) == 2
    assert "000001.SZ" in snap0
    assert "600000.SH" in snap0
    assert "000002.SZ" not in snap0

    # 索引下标访问
    assert 0 in snap0
    assert 1 in snap0
    assert 2 not in snap0
    assert 99 not in snap0

    assert set(snap0.keys()) == {"000001.SZ", "600000.SH"}
    assert len(list(snap0.items())) == 2
    assert len(list(snap0.values())) == 2

    assert snap0["000001.SZ"]["reason"] == "龙虎榜买入"
    assert snap0["600000.SH"]["reason"] == "大宗交易溢价"
    assert snap0["000002.SZ"] is None  # 无事件标的严格返回 None
    assert snap0[2] is None             # 整数索引无事件返回 None

    # Step 1: 仅 600000.SH 有事件
    snap1 = container.get_snapshot(1)
    assert len(snap1) == 1
    assert snap1["000001.SZ"] is None
    assert snap1["600000.SH"]["reason"] == "机构买入"

    # Step 2: 当日无任何事件
    snap2 = container.get_snapshot(2)
    assert bool(snap2) is False
    assert len(snap2) == 0
    assert snap2["000001.SZ"] is None
    assert snap2["600000.SH"] is None
    assert snap2["000002.SZ"] is None


def test_static_attribute_container_comprehensive():
    """测试 StaticAttributeContainer 静态映射容器与未分类返回 None"""
    # 1. 缺失 symbol 异常
    with pytest.raises(ValueError, match="必须包含 'symbol' 列"):
        build_static_container_from_df(pl.DataFrame({"industry": ["银行业"]}), ["000001.SZ"])

    # 2. 单列值模式
    df_single = pl.DataFrame({
        "symbol": ["000001.SZ", "600000.SH"],
        "industry": ["银行业", "银行业"],
    })
    all_symbols = ["000001.SZ", "600000.SH", "000002.SZ"]
    container_single = build_static_container_from_df(df_single, all_symbols)
    assert "StaticAttributeContainer" in repr(container_single)
    assert len(container_single) == 2
    assert "000001.SZ" in container_single
    assert 0 in container_single
    assert "000002.SZ" not in container_single
    assert 2 not in container_single
    assert container_single["000001.SZ"] == "银行业"
    assert container_single[0] == "银行业"
    assert container_single["000002.SZ"] is None  # 未分类标的返回 None
    assert container_single[2] is None
    assert container_single[99] is None

    # 3. 多列字典模式
    df_multi = pl.DataFrame({
        "symbol": ["000001.SZ"],
        "industry": ["银行业"],
        "concept": ["跨境支付"],
    })
    container_multi = build_static_container_from_df(df_multi, all_symbols)
    assert container_multi["000001.SZ"]["industry"] == "银行业"
    assert container_multi["000001.SZ"]["concept"] == "跨境支付"
    assert container_multi["600000.SH"] is None


def test_adj_market_data_explicit_views():
    """测试 AdjMarketData 显式传入价格矩阵的属性访问"""
    mat = np.ones((2, 2)) * 10.0
    adj_data = AdjMarketData(close=mat, open_p=mat, high=mat, low=mat)
    np.testing.assert_allclose(adj_data.close, mat)
    np.testing.assert_allclose(adj_data.open, mat)
    np.testing.assert_allclose(adj_data.high, mat)
    np.testing.assert_allclose(adj_data.low, mat)

    # 既无显式矩阵也无 parent 时抛出 AttributeError
    empty_adj = AdjMarketData()
    with pytest.raises(AttributeError):
        _ = empty_adj.close
