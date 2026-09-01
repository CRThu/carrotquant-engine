"""
单元测试：ColumnDataLoader, ChunkStreamer 与 MarketDataContainer
100% 独立于外部 data/ 路径，完全使用 tmp_path 生成内存与临时测试数据。
"""

import pytest
import numpy as np
import polars as pl
from pathlib import Path

from cq.engine.feed.column_loader import ColumnDataLoader, MarketDataContainer, LazyCustomFields
from cq.engine.feed.chunk_streamer import ChunkStreamer


def test_market_data_container_initialization():
    timestamps = np.array(["2024-01-01", "2024-01-02"])
    symbols = ["000001.SZ", "600000.SH"]

    open_p = np.array([[10.0, 20.0], [10.5, 20.5]])
    high_p = np.array([[10.8, 21.0], [11.0, 21.5]])
    low_p = np.array([[9.9, 19.5], [10.2, 19.8]])
    close_p = np.array([[10.5, 20.2], [10.8, 21.0]])

    container = MarketDataContainer(
        timestamps=timestamps,
        symbols=symbols,
        open_price=open_p,
        high_price=high_p,
        low_price=low_p,
        close_price=close_p,
    )

    assert container.n_steps == 2
    assert container.n_symbols == 2
    assert container.open.flags.c_contiguous
    assert container.close.flags.c_contiguous
    assert np.all(container.is_tradable == True)


def test_column_loader_from_mock_parquet(tmp_path: Path):
    df = pl.DataFrame({
        "symbol": ["000001.SZ", "000001.SZ", "600000.SH", "600000.SH"],
        "datetime": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        "open": [10.0, 10.5, 20.0, 20.5],
        "high": [10.8, 11.0, 21.0, 21.5],
        "low": [9.9, 10.2, 19.5, 19.8],
        "close": [10.5, 10.8, 20.2, 21.0],
        "volume": [1000.0, 1200.0, 5000.0, 5500.0],
        "amount": [10500.0, 12600.0, 101000.0, 115500.0],
    })

    parquet_dir = tmp_path / "kline"
    parquet_dir.mkdir()
    df.write_parquet(parquet_dir / "data.parquet")

    # 1. 基础读取
    container = ColumnDataLoader.load_parquet(path=parquet_dir)
    assert container.n_steps == 2
    assert container.n_symbols == 2
    assert set(container.symbols) == {"000001.SZ", "600000.SH"}
    assert container.close.shape == (2, 2)

    # 2. 带过滤条件读取 (start_date, end_date, symbols, columns)
    adj_df = pl.DataFrame({
        "symbol": ["000001.SZ", "600000.SH"],
        "datetime": ["2024-01-01", "2024-01-01"],
        "back_adj_factor": [1.05, 1.08],
    })
    adj_dir = tmp_path / "kline_adj"
    adj_dir.mkdir()
    adj_df.write_parquet(adj_dir / "adj.parquet")

    filtered = ColumnDataLoader.load_parquet(
        path=parquet_dir,
        adj_factor_path=adj_dir,
        start_date="2024-01-01",
        end_date="2024-01-01",
        symbols=["000001.SZ"],
        columns=["volume"],
    )
    assert filtered.n_steps == 1
    assert filtered.symbols == ["000001.SZ"]
    assert filtered.adj_factor is not None


def test_column_loader_from_mock_csv(tmp_path: Path):
    df = pl.DataFrame({
        "symbol": ["000001.SZ", "000001.SZ", "600000.SH", "600000.SH"],
        "datetime": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        "open": [10.0, 10.5, 20.0, 20.5],
        "high": [10.8, 11.0, 21.0, 21.5],
        "low": [9.9, 10.2, 19.5, 19.8],
        "close": [10.0, 10.0, 20.0, 20.0],
        "volume": [1000.0, 1200.0, 5000.0, 5500.0],
        "amount": [10000.0, 12000.0, 100000.0, 110000.0],
    })

    adj_df = pl.DataFrame({
        "symbol": ["000001.SZ", "000001.SZ", "600000.SH", "600000.SH"],
        "datetime": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        "back_adj_factor": [1.1, 1.1, 1.2, 1.2],
    })

    csv_dir = tmp_path / "csv_kline"
    csv_dir.mkdir()
    csv_file = csv_dir / "stock.csv"
    df.write_csv(csv_file)

    adj_dir = tmp_path / "csv_adj"
    adj_dir.mkdir()
    adj_file = adj_dir / "adj.csv"
    adj_df.write_csv(adj_file)

    # 1. 目录模式
    container = ColumnDataLoader.load_csv(path=csv_dir, adj_factor_path=adj_dir)
    assert container.n_steps == 2
    assert container.n_symbols == 2
    assert container.close[0, 0] == 10.0
    assert container.adj.close[0, 0] == 11.0

    # 2. 单文件模式与过滤
    single_res = ColumnDataLoader.load_csv(
        path=csv_file,
        adj_factor_path=adj_file,
        start_date="2024-01-01",
        end_date="2024-01-01",
        symbols=["000001.SZ"],
    )
    assert single_res.n_steps == 1
    assert single_res.symbols == ["000001.SZ"]


def test_scan_parquet_chunks_partition_modes(tmp_path: Path):
    """测试 scan_parquet_chunks 支持 year/month/default 分区模式"""
    df = pl.DataFrame({
        "symbol": ["000001.SZ", "000001.SZ"],
        "datetime": ["2023-12-31", "2024-01-01"],
        "open": [10.0, 10.5],
        "high": [10.8, 11.0],
        "low": [9.9, 10.2],
        "close": [10.5, 10.8],
        "volume": [1000.0, 1200.0],
        "amount": [10500.0, 12600.0],
    })
    p_dir = tmp_path / "chunk_kline"
    p_dir.mkdir()
    df.write_parquet(p_dir / "data.parquet")

    # 1. 按年
    year_chunks = list(ColumnDataLoader.scan_parquet_chunks(p_dir, partition_by="year"))
    assert len(year_chunks) == 2

    # 2. 默认模式
    default_chunks = list(ColumnDataLoader.scan_parquet_chunks(p_dir, partition_by="none"))
    assert len(default_chunks) == 1


def test_lazy_custom_fields_standalone():
    """测试 LazyCustomFields 独立透视与 KeyError 拦截"""
    df = pl.DataFrame({
        "symbol": ["000001.SZ", "600000.SH"],
        "datetime": ["2024-01-01", "2024-01-01"],
        "alpha_01": [0.5, -0.5],
    })
    lcf = LazyCustomFields(df=df, all_timestamps=["2024-01-01"], all_symbols=["000001.SZ", "600000.SH"])
    assert "alpha_01" in lcf
    assert "beta" not in lcf
    assert "alpha_01" in lcf.keys()

    mat = lcf["alpha_01"]
    assert mat.shape == (1, 2)
    np.testing.assert_allclose(mat[0], [0.5, -0.5])

    # 缓存读取
    mat_cached = lcf.get("alpha_01")
    assert mat_cached is mat

    with pytest.raises(KeyError, match="不存在"):
        lcf.get("unknown")


def test_chunk_streamer():
    timestamps = np.array([f"2024-01-{i+1:02d}" for i in range(10)])
    symbols = ["000001.SZ"]
    prices = np.ones((10, 1))

    container = MarketDataContainer(
        timestamps=timestamps,
        symbols=symbols,
        open_price=prices,
        high_price=prices,
        low_price=prices,
        close_price=prices,
        volume=prices * 100,
        amount=prices * 1000,
    )

    streamer = ChunkStreamer(container, chunk_size=3)
    chunks = list(streamer.iter_chunks())

    assert len(chunks) == 4
    start_idx, end_idx, chunk = chunks[0]
    assert start_idx == 0
    assert end_idx == 3
    assert chunk.n_steps == 3
    assert chunk.volume.shape == (3, 1)
