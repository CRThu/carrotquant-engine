"""
列式按需加载器 (ColumnDataLoader)

负责从 Parquet (Hive 分区结构) 或 CSV 文件中按列加载数据，
以 Int64 毫秒时间戳 (timestamp) 为唯一权威事实基准，
利用 Polars 进行时间轴与标的池的极速对齐，构建 C-Contiguous 的 2D NumPy 矩阵块。
"""

from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import polars as pl

from cq.engine.feed.matrix_builder import (
    MarketData,
    MarketDataContainer,
    AdjMarketData,
    LazyCustomFields,
    SparseEventContainer,
    StaticAttributeContainer,
    build_market_data_from_df,
)
from cq.engine.utils.time_utils import parse_date_to_ms


class ColumnDataLoader:
    """
    列式行情与复权因子按需加载器 (支持 Parquet 与 CSV 格式的数据源)
    """

    @classmethod
    def load_parquet(
        cls,
        path: Union[str, Path],
        adj_factor_path: Optional[Union[str, Path]] = None,
        columns: Optional[List[str]] = None,
        custom_columns: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
    ) -> MarketData:
        """
        从 Parquet 目录装载全量行情矩阵 (优先以 timestamp Int64 为核心坐标)。
        """
        path = Path(path)
        base_cols = {"symbol", "timestamp", "datetime", "open", "high", "low", "close", "volume", "amount"}

        df = pl.scan_parquet(str(path / "**" / "*.parquet"))
        schema_names = df.collect_schema().names()

        target_columns = columns or custom_columns
        requested_custom = set(target_columns or [])
        if requested_custom:
            target_cols = list(base_cols | requested_custom)
        else:
            target_cols = schema_names

        # 优先使用 Int64 timestamp 进行高阶向量化数值过滤
        if "timestamp" in schema_names:
            if start_date:
                df = df.filter(pl.col("timestamp") >= parse_date_to_ms(start_date))
            if end_date:
                df = df.filter(pl.col("timestamp") <= parse_date_to_ms(end_date))
        elif "datetime" in schema_names:
            if start_date:
                df = df.filter(pl.col("datetime") >= start_date)
            if end_date:
                df = df.filter(pl.col("datetime") <= end_date)

        if symbols and "symbol" in schema_names:
            df = df.filter(pl.col("symbol").is_in(symbols))

        available_cols = [col for col in target_cols if col in schema_names]
        df_collected = df.select(available_cols).collect()

        if df_collected.is_empty():
            raise ValueError(f"No parquet data found in {path} with specified filters.")

        # 复权因子按 [symbol, timestamp] Join
        if adj_factor_path:
            adj_path = Path(adj_factor_path)
            adj_df = pl.scan_parquet(str(adj_path / "**" / "*.parquet"))
            adj_schema = adj_df.collect_schema().names()

            time_key = "timestamp" if "timestamp" in adj_schema and "timestamp" in df_collected.columns else "datetime"
            if time_key == "timestamp":
                if start_date:
                    adj_df = adj_df.filter(pl.col("timestamp") >= parse_date_to_ms(start_date))
                if end_date:
                    adj_df = adj_df.filter(pl.col("timestamp") <= parse_date_to_ms(end_date))
            else:
                if start_date:
                    adj_df = adj_df.filter(pl.col("datetime") >= start_date)
                if end_date:
                    adj_df = adj_df.filter(pl.col("datetime") <= end_date)

            if symbols and "symbol" in adj_schema:
                adj_df = adj_df.filter(pl.col("symbol").is_in(symbols))
            adj_df = adj_df.select(["symbol", time_key, "back_adj_factor"]).collect()

            df_collected = df_collected.join(adj_df, on=["symbol", time_key], how="left")

        return cls._build_container_from_df(df_collected, columns=target_columns)

    @classmethod
    def scan_parquet_chunks(
        cls,
        path: Union[str, Path],
        adj_factor_path: Optional[Union[str, Path]] = None,
        columns: Optional[List[str]] = None,
        custom_columns: Optional[List[str]] = None,
        partition_by: str = "year",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
    ) -> Generator[MarketData, None, None]:
        """
        从 Parquet 目录按 Hive 分区 (如按年 year 或按月 month) 惰性分块生成器。
        以 Int64 毫秒时间戳为核心投影分块键，杜绝大字符串全量扫描。
        """
        path = Path(path)
        scan_lazy = pl.scan_parquet(str(path / "**" / "*.parquet"))
        schema_names = scan_lazy.collect_schema().names()
        has_ts = "timestamp" in schema_names

        if has_ts:
            if start_date:
                scan_lazy = scan_lazy.filter(pl.col("timestamp") >= parse_date_to_ms(start_date))
            if end_date:
                scan_lazy = scan_lazy.filter(pl.col("timestamp") <= parse_date_to_ms(end_date))
        elif "datetime" in schema_names:
            if start_date:
                scan_lazy = scan_lazy.filter(pl.col("datetime") >= start_date)
            if end_date:
                scan_lazy = scan_lazy.filter(pl.col("datetime") <= end_date)

        if symbols and "symbol" in schema_names:
            scan_lazy = scan_lazy.filter(pl.col("symbol").is_in(symbols))

        target_columns = columns or custom_columns

        # 提取起止分区键
        if partition_by == "year":
            if has_ts:
                years = (
                    scan_lazy.select(pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.year().alias("k"))
                    .unique()
                    .sort("k")
                    .collect()["k"]
                    .to_list()
                )
            else:
                years = sorted(list(set(d[:4] for d in scan_lazy.select(pl.col("datetime")).collect()["datetime"])))

            for y in years:
                sub_start = f"{y}-01-01"
                sub_end = f"{y}-12-31 23:59:59"
                yield cls.load_parquet(
                    path=path,
                    adj_factor_path=adj_factor_path,
                    columns=target_columns,
                    start_date=sub_start,
                    end_date=sub_end,
                    symbols=symbols,
                )
        elif partition_by == "month":
            if has_ts:
                months = (
                    scan_lazy.select(
                        pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.strftime("%Y-%m").alias("k")
                    )
                    .unique()
                    .sort("k")
                    .collect()["k"]
                    .to_list()
                )
            else:
                months = sorted(list(set(d[:7] for d in scan_lazy.select(pl.col("datetime")).collect()["datetime"])))

            for m in months:
                sub_start = f"{m}-01"
                sub_end = f"{m}-31 23:59:59"
                yield cls.load_parquet(
                    path=path,
                    adj_factor_path=adj_factor_path,
                    columns=target_columns,
                    start_date=sub_start,
                    end_date=sub_end,
                    symbols=symbols,
                )
        else:
            yield cls.load_parquet(
                path=path,
                adj_factor_path=adj_factor_path,
                columns=target_columns,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
            )

    @classmethod
    def load_csv(
        cls,
        path: Union[str, Path],
        adj_factor_path: Optional[Union[str, Path]] = None,
        columns: Optional[List[str]] = None,
        custom_columns: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
    ) -> MarketData:
        """
        从 CSV 目录或文件装载行情矩阵 (优先使用 timestamp Int64)。
        """
        path = Path(path)
        base_cols = {"symbol", "timestamp", "datetime", "open", "high", "low", "close", "volume", "amount"}

        if path.is_file():
            scan_path = str(path)
        else:
            scan_path = str(path / "**" / "*.csv")

        df = pl.scan_csv(scan_path, infer_schema_length=10000)
        schema_names = df.collect_schema().names()

        target_columns = columns or custom_columns
        requested_custom = set(target_columns or [])
        if requested_custom:
            target_cols = list(base_cols | requested_custom)
        else:
            target_cols = schema_names

        has_ts = "timestamp" in schema_names
        if has_ts:
            if start_date:
                df = df.filter(pl.col("timestamp") >= parse_date_to_ms(start_date))
            if end_date:
                df = df.filter(pl.col("timestamp") <= parse_date_to_ms(end_date))
        elif "datetime" in schema_names:
            if start_date:
                df = df.filter(pl.col("datetime") >= start_date)
            if end_date:
                df = df.filter(pl.col("datetime") <= end_date)

        if symbols and "symbol" in schema_names:
            df = df.filter(pl.col("symbol").is_in(symbols))

        available_cols = [col for col in target_cols if col in schema_names]
        df_collected = df.select(available_cols).collect()

        if df_collected.is_empty():
            raise ValueError(f"No CSV data found in {path} with specified filters.")

        if adj_factor_path:
            adj_path = Path(adj_factor_path)
            adj_scan = str(adj_path) if adj_path.is_file() else str(adj_path / "**" / "*.csv")
            adj_df = pl.scan_csv(adj_scan, infer_schema_length=10000)
            adj_schema = adj_df.collect_schema().names()
            time_key = "timestamp" if "timestamp" in adj_schema and "timestamp" in df_collected.columns else "datetime"

            if time_key == "timestamp":
                if start_date:
                    adj_df = adj_df.filter(pl.col("timestamp") >= parse_date_to_ms(start_date))
                if end_date:
                    adj_df = adj_df.filter(pl.col("timestamp") <= parse_date_to_ms(end_date))
            else:
                if start_date:
                    adj_df = adj_df.filter(pl.col("datetime") >= start_date)
                if end_date:
                    adj_df = adj_df.filter(pl.col("datetime") <= end_date)

            if symbols and "symbol" in adj_schema:
                adj_df = adj_df.filter(pl.col("symbol").is_in(symbols))
            adj_df = adj_df.select(["symbol", time_key, "back_adj_factor"]).collect()

            df_collected = df_collected.join(adj_df, on=["symbol", time_key], how="left")

        return cls._build_container_from_df(df_collected, columns=target_columns)

    @classmethod
    def _build_container_from_df(
        cls,
        df: pl.DataFrame,
        columns: Optional[List[str]] = None,
        custom_columns: Optional[List[str]] = None,
    ) -> MarketData:
        """从 Polars DataFrame 单趟极速构建 2D 对齐矩阵与 MarketData"""
        target_columns = columns or custom_columns
        return build_market_data_from_df(df, columns=target_columns)
