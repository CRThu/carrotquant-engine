"""
单元测试：时间工具模块 (test_time_utils.py)
"""

import pytest
from datetime import datetime, timezone
import polars as pl

from cq.engine.utils.time_utils import (
    parse_date_to_ms,
    ts_to_iso_str,
    extract_timestamps_series,
)


def test_parse_date_to_ms_various_formats():
    """测试 parse_date_to_ms 对各种输入格式的解析"""
    # 1. None 与空串
    assert parse_date_to_ms(None) == 0
    assert parse_date_to_ms("") == 0
    assert parse_date_to_ms("   ") == 0

    # 2. 整数 / 浮点数毫秒戳
    assert parse_date_to_ms(1704067200000) == 1704067200000
    assert parse_date_to_ms(1704067200000.0) == 1704067200000
    assert parse_date_to_ms("1704067200000") == 1704067200000

    # 3. YYYY-MM-DD
    ms_day = parse_date_to_ms("2024-01-01")
    assert ms_day == 1704067200000

    # 4. YYYY-MM-DD HH:MM:SS
    ms_dt = parse_date_to_ms("2024-01-01 15:00:00")
    assert ms_dt > 1704067200000

    # 5. ISO8601 带时区
    ms_iso = parse_date_to_ms("2024-01-01T15:00:00.000+08:00")
    assert ms_iso > 0

    # 6. 非法格式
    with pytest.raises(ValueError, match="不支持的日期格式"):
        parse_date_to_ms([12345])


def test_ts_to_iso_str():
    """测试 ts_to_iso_str 转换与异常保底"""
    assert ts_to_iso_str(None) == ""
    assert ts_to_iso_str(0) == ""
    assert ts_to_iso_str(-1) == ""

    # 1704067200000 -> 2024-01-01 00:00:00 UTC
    res = ts_to_iso_str(1704067200000)
    assert res == "2024-01-01 00:00:00"


def test_extract_timestamps_series():
    """测试 extract_timestamps_series 对 timestamp / datetime / Date 等列的处理"""
    # 1. 含有 timestamp (Int64)
    df_ts = pl.DataFrame({"timestamp": [1704067200000, 1704153600000]})
    s1 = extract_timestamps_series(df_ts)
    assert s1.dtype == pl.Int64
    assert s1.to_list() == [1704067200000, 1704153600000]

    # 2. 含有 datetime (String)
    df_str = pl.DataFrame({"datetime": ["2024-01-01", "2024-01-02"]})
    s2 = extract_timestamps_series(df_str)
    assert s2.dtype == pl.Int64
    assert s2.to_list() == [1704067200000, 1704153600000]

    # 3. 含有 datetime (Datetime)
    df_dt = pl.DataFrame({"datetime": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"]}).with_columns(
        pl.col("datetime").str.to_datetime(time_unit="ms")
    )
    s3 = extract_timestamps_series(df_dt)
    assert s3.dtype == pl.Int64
    assert s3.to_list() == [1704067200000, 1704153600000]

    # 4. 缺失时间列抛出异常
    df_invalid = pl.DataFrame({"value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="数据源缺失必需的时间列"):
        extract_timestamps_series(df_invalid)
