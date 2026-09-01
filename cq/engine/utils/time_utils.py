"""
时间戳与时区转换工具模块 (cq.engine.utils.time_utils)

提供 UTC 毫秒级时间戳 (Int64) 与日期字符串、ISO 格式以及 Polars Series 之间的单点权威转换。
严格基于标准库 datetime 与 timezone.utc，保持零重量级第三方依赖。
"""

from typing import Any, Optional, Union
from datetime import datetime, timezone
import numpy as np
import polars as pl


def parse_date_to_ms(date_val: Any) -> int:
    """
    将输入日期 (yyyy-mm-dd 字符串、ISO 字符串或毫秒级整数) 解析为 UTC 0 毫秒级 Int64 时间戳。
    """
    if date_val is None:
        return 0
    if isinstance(date_val, (int, float, np.integer, np.floating)):
        return int(date_val)
    if isinstance(date_val, str):
        clean_str = date_val.strip()
        if not clean_str:
            return 0
        try:
            return int(clean_str)
        except ValueError:
            pass

        clean_iso = clean_str.replace(" ", "T")
        if len(clean_iso) == 10:  # YYYY-MM-DD
            dt = datetime.strptime(clean_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            if "+" not in clean_iso and "Z" not in clean_iso:
                clean_iso += "+00:00"
            dt = datetime.fromisoformat(clean_iso)
        return int(dt.timestamp() * 1000)

    raise ValueError(f"不支持的日期格式：{date_val} (类型: {type(date_val)})")


def ts_to_iso_str(ts_ms: Any, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    将 UTC 毫秒时间戳格式化为可读字符串 (供日志与策略调试展示)。
    若已是字符串则直接返回。
    """
    if ts_ms is None:
        return ""
    if isinstance(ts_ms, (str, np.str_)):
        return str(ts_ms)
    if isinstance(ts_ms, (int, float, np.integer, np.floating)):
        if ts_ms <= 0:
            return ""
        try:
            dt = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
            return dt.strftime(fmt)
        except (ValueError, OSError, OverflowError):
            return str(ts_ms)
    return str(ts_ms)


def extract_timestamps_series(df: pl.DataFrame) -> pl.Series:
    """
    从 Polars DataFrame 中提取并标准化为 UTC 毫秒级 Int64 时间戳 Series。
    优先以 'timestamp' (Int64) 为权威主键直读；若仅有 'datetime' 字符串/日期列则执行无损转换。
    """
    if "timestamp" in df.columns:
        return df["timestamp"].cast(pl.Int64)
    elif "datetime" in df.columns:
        dt_col = df["datetime"]
        if dt_col.dtype in (pl.Datetime, pl.Date):
            return dt_col.cast(pl.Datetime("ms")).cast(pl.Int64)
        elif dt_col.dtype == pl.String:
            parsed = dt_col.str.to_datetime(time_unit="ms", strict=False)
            if parsed.null_count() == len(df):
                parsed = dt_col.str.to_datetime(format="%Y-%m-%d", time_unit="ms", strict=False)
            return parsed.cast(pl.Int64)
        else:
            try:
                return dt_col.cast(pl.Datetime("ms")).cast(pl.Int64)
            except Exception:
                str_list = [str(x) for x in dt_col.to_list()]
                return pl.Series(str_list).str.to_datetime(time_unit="ms", strict=False).cast(pl.Int64)
    else:
        raise ValueError("数据源缺失必需的时间列：'timestamp' 或 'datetime'")
