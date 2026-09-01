"""
单元测试：统一数据供给解析器 (test_feed_loader.py)
"""

import pytest
import numpy as np
import polars as pl

from cq.engine.feed.feed_loader import (
    load_feed,
    stream_feed,
    resolve_duck_df,
    ts_table,
    event_table,
    static_table,
)
from cq.engine.feed.matrix_builder import (
    MarketData,
    TimeSeriesTable,
    SparseEventContainer,
    StaticAttributeContainer,
    build_market_data_from_df,
    build_event_container_from_df,
    build_static_container_from_df,
)


class MockQueryWithToDf:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def to_df(self) -> pl.DataFrame:
        return self._df


class MockQueryWithRead:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def read(self) -> pl.DataFrame:
        return self._df


class MockStreamObject:
    def __init__(self, chunks):
        self._chunks = chunks

    def iter_chunks(self):
        for i, chunk in enumerate(self._chunks):
            yield i * 10, (i + 1) * 10, chunk


def test_duck_typing_resolution():
    """测试 Duck Typing 协议解构 (to_df, read, collect, LazyFrame, DataFrame, Callable)"""
    df = pl.DataFrame({
        "timestamp": [1704067200000],
        "symbol": ["000001.SZ"],
        "open": [10.0],
        "high": [11.0],
        "low": [9.5],
        "close": [10.2],
    })

    # 1. to_df 鸭子对象
    res1 = resolve_duck_df(MockQueryWithToDf(df))
    assert isinstance(res1, pl.DataFrame)

    # 2. read 鸭子对象
    res2 = resolve_duck_df(MockQueryWithRead(df))
    assert isinstance(res2, pl.DataFrame)

    # 3. LazyFrame .collect()
    res3 = resolve_duck_df(df.lazy())
    assert isinstance(res3, pl.DataFrame)

    # 4. 原生 DataFrame
    res4 = resolve_duck_df(df)
    assert isinstance(res4, pl.DataFrame)

    # 5. 无参 Callable 闭包 (直接返回 df 或返回 LazyFrame)
    res5 = resolve_duck_df(lambda: df)
    assert isinstance(res5, pl.DataFrame)
    res5_lazy = resolve_duck_df(lambda: df.lazy())
    assert isinstance(res5_lazy, pl.DataFrame)

    # 6. 直接透传 MarketData
    mdata = build_market_data_from_df(df)
    res6 = resolve_duck_df(mdata)
    assert res6 is mdata

    # 7. 不支持的非法类型
    with pytest.raises(TypeError, match="无法识别的数据供给对象类型"):
        resolve_duck_df(12345)


def test_load_feed_single_table_shorthand():
    """测试单表直接传入简写 (自动封装为 stock 主表)"""
    df = pl.DataFrame({
        "timestamp": [1704067200000, 1704153600000],
        "symbol": ["000001.SZ", "000001.SZ"],
        "open": [10.0, 10.5],
        "high": [11.0, 11.5],
        "low": [9.5, 10.0],
        "close": [10.2, 10.8],
    })

    data = load_feed(df, symbols=["000001.SZ"])
    assert isinstance(data, MarketData)
    assert data.shape == (2, 1)
    assert data.symbols == ["000001.SZ"]


def test_load_feed_empty_dict_and_invalid_main():
    """测试空字典和非法主表异常拦截"""
    with pytest.raises(ValueError, match="传入的 data 字典为空"):
        load_feed({})

    with pytest.raises(TypeError, match="无法识别的数据供给对象类型"):
        load_feed({"stock": 12345})


def test_load_feed_unannotated_secondary_raises_error():
    """测试副表未显式声明类型时抛出明确 TypeError 杜绝黑盒误判"""
    main_df = pl.DataFrame({
        "timestamp": [1704067200000],
        "symbol": ["000001.SZ"],
        "open": [10.0],
        "high": [11.0],
        "low": [9.5],
        "close": [10.2],
    })
    sec_df = pl.DataFrame({
        "timestamp": [1704067200000],
        "symbol": ["000001.SZ"],
        "pe": [12.0],
    })

    with pytest.raises(TypeError, match="副表 'valuation' 未显式声明表类型"):
        load_feed({"stock": main_df, "valuation": sec_df})


def test_load_feed_master_clock_multitable_alignment():
    """测试 Master Clock 主时钟多表对齐机制与显式规约 (ts_table / event_table / static_table / tuple)"""
    main_df = pl.DataFrame({
        "timestamp": [1704067200000, 1704153600000, 1704067200000, 1704153600000],
        "symbol": ["000001.SZ", "000001.SZ", "600000.SH", "600000.SH"],
        "open": [10.0, 10.5, 20.0, 20.5],
        "high": [11.0, 11.5, 21.0, 21.5],
        "low": [9.5, 10.0, 19.5, 20.0],
        "close": [10.2, 10.8, 20.2, 20.8],
    })

    index_df = pl.DataFrame({
        "timestamp": [1704067200000, 1704153600000, 1704240000000],
        "symbol": ["000001.SZ", "000001.SZ", "000001.SZ"],
        "open": [3000.0, 3050.0, 3100.0],
        "high": [3050.0, 3100.0, 3150.0],
        "low": [2980.0, 3020.0, 3080.0],
        "close": [3020.0, 3080.0, 3120.0],
    })

    # 多因子时序表 (pe_ttm, pb, roe)
    val_df = pl.DataFrame({
        "timestamp": [1704067200000, 1704153600000],
        "symbol": ["000001.SZ", "000001.SZ"],
        "pe_ttm": [12.5, 13.0],
        "pb": [1.1, 1.2],
    })

    # 离散事件表 (龙虎榜)
    dt_df = pl.DataFrame({
        "timestamp": [1704067200000],
        "symbol": ["000001.SZ"],
        "reason": ["机构净买入"],
        "net_amt": [10000000.0],
    })

    # 静态属性表 (行业/概念板块，含多行平铺)
    industry_df = pl.DataFrame({
        "symbol": ["000001.SZ", "000001.SZ"],
        "concept": ["信创", "跨境支付"],
    })

    data_bundle = load_feed({
        "stock": main_df,
        "index": ts_table(index_df),
        "valuation": (val_df, "ts"),  # 支持元组快捷方式
        "dragon_tiger": event_table(dt_df),
        "industry": static_table(industry_df),
    })

    assert isinstance(data_bundle, MarketData)
    assert data_bundle.shape == (2, 2)
    assert "index" in data_bundle.tables
    assert "valuation" in data_bundle.tables
    assert "dragon_tiger" in data_bundle.tables
    assert "industry" in data_bundle.tables

    # 验证副行情表 MarketData
    index_tbl = data_bundle.tables["index"]
    assert isinstance(index_tbl, MarketData)
    assert index_tbl.shape == (2, 2)
    np.testing.assert_allclose(index_tbl.close[:, 0], [3020.0, 3080.0])
    assert np.isnan(index_tbl.close[:, 1]).all()

    # 验证多因子 TimeSeriesTable
    val_tbl = data_bundle.tables["valuation"]
    assert isinstance(val_tbl, TimeSeriesTable)
    assert val_tbl.shape == (2, 2)
    np.testing.assert_allclose(val_tbl.pe_ttm[:, 0], [12.5, 13.0])
    np.testing.assert_allclose(val_tbl.pb[:, 0], [1.1, 1.2])

    # 验证静态属性一对多列表聚合
    ind_tbl = data_bundle.tables["industry"]
    assert isinstance(ind_tbl, StaticAttributeContainer)
    assert ind_tbl["000001.SZ"] == ["信创", "跨境支付"]


def test_load_feed_with_prebuilt_containers():
    """测试多表中直接传入已构建的 SparseEventContainer 与 StaticAttributeContainer 实例"""
    main_df = pl.DataFrame({
        "timestamp": [1704067200000, 1704153600000],
        "symbol": ["000001.SZ", "000001.SZ"],
        "open": [10.0, 10.5],
        "high": [11.0, 11.5],
        "low": [9.5, 10.0],
        "close": [10.2, 10.8],
    })

    ts_arr = np.array([1704067200000, 1704153600000], dtype=np.int64)
    sym_list = ["000001.SZ"]

    dt_df = pl.DataFrame({
        "timestamp": [1704067200000],
        "symbol": ["000001.SZ"],
        "reason": ["龙虎榜"],
    })
    prebuilt_event = build_event_container_from_df(dt_df, ts_arr, sym_list)

    ind_df = pl.DataFrame({
        "symbol": ["000001.SZ"],
        "concept": ["信创"],
    })
    prebuilt_static = build_static_container_from_df(ind_df, sym_list)

    bundle = load_feed({
        "stock": main_df,
        "dt": prebuilt_event,
        "concepts": prebuilt_static,
    })

    assert bundle.tables["dt"] is prebuilt_event
    assert bundle.tables["concepts"] is prebuilt_static


def test_stream_feed_generator_and_chunks():
    """测试 stream_feed 分块流式生成器与 iter_chunks 鸭子对象"""
    df = pl.DataFrame({
        "timestamp": [1704067200000 + i * 86400000 for i in range(10)],
        "symbol": ["000001.SZ"] * 10,
        "open": [10.0 + i for i in range(10)],
        "high": [11.0 + i for i in range(10)],
        "low": [9.5 + i for i in range(10)],
        "close": [10.2 + i for i in range(10)],
    })

    # 1. 传入 MarketData
    data = load_feed(df)
    chunks1 = list(stream_feed(data))
    assert len(chunks1) == 1

    # 2. 传入带有 iter_chunks 方法的对象
    stream_obj = MockStreamObject([df, df])
    chunks2 = list(stream_feed(stream_obj))
    assert len(chunks2) == 2

    # 3. 传入生成器 / Iterator
    def gen_chunks():
        yield df
        yield df

    chunks3 = list(stream_feed(gen_chunks()))
    assert len(chunks3) == 2
