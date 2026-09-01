"""
统一数据供给解析器 (FeedLoader)

负责解构遵循 Duck Typing 协议的多种输入源（Query 鸭子对象、LazyFrame、DataFrame、Stream 及函数闭包），
执行显式表类型规约、TS 多表 Master Clock 网格对齐、离散 Event 稀疏索引构建与分块流式加载。
"""

from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union
import numpy as np
import polars as pl

from cq.engine.feed.matrix_builder import (
    build_market_data_from_df,
    build_timeseries_table_from_df,
    build_event_container_from_df,
    build_static_container_from_df,
    MarketData,
    TimeSeriesTable,
    SparseEventContainer,
    StaticAttributeContainer,
)


class TableSpec:
    """
    显式数据表类型规约 (TableSpec)
    用于明确副表的物理语义 (ts 时序表 / event 离散事件表 / static 静态属性表)，
    彻底杜绝黑盒误判与模糊猜测。
    """

    def __init__(self, data: Any, table_type: str, columns: Optional[List[str]] = None):
        self.data = data
        self.table_type = table_type.lower()
        self.columns = columns
        if self.table_type not in ("ts", "event", "static"):
            raise ValueError(f"不支持的表类型: '{table_type}'。有效选项为: 'ts', 'event', 'static'")

    def __repr__(self) -> str:
        return f"<TableSpec: type='{self.table_type}', data={type(self.data)}>"


def ts_table(data: Any, columns: Optional[List[str]] = None) -> TableSpec:
    """显式声明为时序表 (TS Table: 行情或多因子特征表，各数值列对齐为 (T, N) 2D 矩阵)"""
    return TableSpec(data, "ts", columns=columns)


def event_table(data: Any) -> TableSpec:
    """显式声明为离散事件表 (Event Table: 龙虎榜、大宗交易等，按时间戳稀疏索引)"""
    return TableSpec(data, "event")


def static_table(data: Any) -> TableSpec:
    """显式声明为静态属性表 (Static Table: 行业分类、概念板块等无时间轴维度映射)"""
    return TableSpec(data, "static")


def resolve_duck_df(val: Any) -> Any:
    """
    遵循纯净 Duck Typing 协议解构输入对象：
    1. MarketData / Container: 直接透传；
    2. Query/Reader 鸭子对象: 调用 .to_df() 或 .read()；
    3. LazyFrame: 调用 .collect()；
    4. DataFrame: 原生 pl.DataFrame；
    5. 无参 Callable: 闭包执行；
    6. Stream / Iterator: 保留生成器。
    """
    if isinstance(val, (MarketData, TimeSeriesTable, SparseEventContainer, StaticAttributeContainer)):
        return val

    if hasattr(val, "to_df") and callable(getattr(val, "to_df")):
        return val.to_df()
    elif hasattr(val, "read") and callable(getattr(val, "read")):
        return val.read()
    elif hasattr(val, "collect") and callable(getattr(val, "collect")):
        return val.collect()
    elif isinstance(val, pl.DataFrame):
        return val
    elif isinstance(val, pl.LazyFrame):
        return val.collect()
    elif callable(val):
        res = val()
        if isinstance(res, pl.LazyFrame):
            return res.collect()
        return res
    elif hasattr(val, "iter_chunks") or (isinstance(val, (Iterable, Iterator, Generator)) and not isinstance(val, (dict, list, tuple, str))):
        return val
    else:
        raise TypeError(
            f"无法识别的数据供给对象类型: {type(val)}。"
            f"遵循 Duck Typing 协议，必须具备 .to_df(), .read(), .collect() 方法或为 DataFrame/Stream。"
        )


def load_feed(
    data: Union[dict, Any],
    main_key: Optional[str] = None,
    columns: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
) -> MarketData:
    """
    统一数据供给装载入口 (load_feed)

    支持：
    1. 单表简写（DataFrame / Duck 对象直接传入，内部自动封装为 {"stock": data}）；
    2. 多表字典（包含主行情表、副行情表、多因子特征表、离散事件表、静态属性表）；
    3. 严格显式指定副表语义 (ts_table / event_table / static_table)，严禁隐式猜测；
    4. 自动以 Master Clock（主时钟）为基准对齐副表与标的池。
    """
    if not isinstance(data, dict):
        dict_data = {"stock": data}
        effective_main_key = "stock"
    else:
        dict_data = dict(data)
        if len(dict_data) == 0:
            raise ValueError("传入的 data 字典为空，无法装载 Feed。")
        effective_main_key = main_key if (main_key and main_key in dict_data) else next(iter(dict_data.keys()))

    # 1. 解析主表并锁定 Master Clock
    main_entry = dict_data[effective_main_key]
    if isinstance(main_entry, TableSpec):
        main_entry = main_entry.data
    elif isinstance(main_entry, tuple) and len(main_entry) == 2:
        main_entry = main_entry[0]

    main_raw = resolve_duck_df(main_entry)
    if isinstance(main_raw, MarketData):
        main_market_data = main_raw
    elif isinstance(main_raw, pl.DataFrame):
        if symbols:
            main_raw = main_raw.filter(pl.col("symbol").is_in(symbols))
        main_market_data = build_market_data_from_df(main_raw, is_main_clock=True, columns=columns)
    else:
        raise TypeError(f"主行情表 '{effective_main_key}' 无法解析为 MarketData 或 DataFrame: {type(main_raw)}")

    master_ts = main_market_data.timestamps
    master_syms = main_market_data.symbols

    # 2. 解析副表并按 Master Clock 内存对齐与分类
    secondary_tables: Dict[str, Any] = {}
    for k, v in dict_data.items():
        if k == effective_main_key:
            continue

        # 显式规约或已实例化容器解析
        if isinstance(v, (MarketData, TimeSeriesTable, SparseEventContainer, StaticAttributeContainer)):
            secondary_tables[k] = v
            continue

        table_type: Optional[str] = None
        table_data: Any = v
        table_cols: Optional[List[str]] = columns

        if isinstance(v, TableSpec):
            table_type = v.table_type
            table_data = v.data
            table_cols = v.columns or columns
        elif isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], str):
            table_data = v[0]
            table_type = v[1].lower()
        else:
            raise TypeError(
                f"副表 '{k}' 未显式声明表类型。为了防止量化回测产生黑盒误判与模糊推测，"
                f"副表必须通过显式声明 (如 cq.engine.ts_table({k}), cq.engine.event_table({k}), "
                f"cq.engine.static_table({k}) 或 ({k}, 'ts'|'event'|'static')) 明确物理语义。"
            )

        resolved = resolve_duck_df(table_data)
        if not isinstance(resolved, pl.DataFrame):
            raise TypeError(f"副表 '{k}' 解构后必须为 Polars DataFrame，实际为: {type(resolved)}")

        if table_type == "ts":
            # 时序表 (TS Table: 副行情或多因子特征表 -> 对齐为 2D 矩阵)
            secondary_tables[k] = build_timeseries_table_from_df(
                resolved,
                all_timestamps=master_ts,
                all_symbols=master_syms,
                columns=table_cols,
            )
        elif table_type == "event":
            # 离散事件表 (Event Table: 龙虎榜、大宗交易等 -> 稀疏时序索引)
            secondary_tables[k] = build_event_container_from_df(
                resolved,
                all_timestamps=master_ts,
                all_symbols=master_syms,
            )
        elif table_type == "static":
            # 静态属性表 (Static Table: 行业、概念板块等 -> 标的维度映射)
            secondary_tables[k] = build_static_container_from_df(
                resolved,
                all_symbols=master_syms,
            )
        else:
            raise ValueError(f"未知的表类型: '{table_type}'，仅支持 'ts', 'event', 'static'")

    # 3. 将副表与事件容器挂载至主 MarketData.tables
    main_market_data.tables.update(secondary_tables)
    return main_market_data


def stream_feed(
    data: Union[dict, Any],
    partition_by: str = "year",
    warmup_steps: int = 0,
    columns: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
) -> Generator[MarketData, None, None]:
    """
    分块流式数据供给生成器 (stream_feed)
    支持从具备 iter_chunks() 的流式对象或分块生成器中按需产生 MarketData 切片。
    """
    if hasattr(data, "iter_chunks") and callable(getattr(data, "iter_chunks")):
        for chunk in data.iter_chunks():
            if isinstance(chunk, tuple) and len(chunk) == 3:
                _, _, c_data = chunk
                yield load_feed(c_data, columns=columns, symbols=symbols)
            else:
                yield load_feed(chunk, columns=columns, symbols=symbols)
    elif isinstance(data, (Iterable, Iterator, Generator)) and not isinstance(data, (dict, list, tuple, str, pl.DataFrame, MarketData)):
        for chunk in data:
            yield load_feed(chunk, columns=columns, symbols=symbols)
    elif isinstance(data, MarketData):
        from cq.engine.feed.chunk_streamer import ChunkStreamer
        streamer = ChunkStreamer(data, chunk_size=1000)
        for _, _, chunk in streamer.iter_chunks():
            yield chunk
    else:
        # 单块数据
        yield load_feed(data, columns=columns, symbols=symbols)
