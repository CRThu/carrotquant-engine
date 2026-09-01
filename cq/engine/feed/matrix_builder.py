"""
单趟极速矩阵构建器与数据容器体系 (cq.engine.feed.matrix_builder)

负责从 Polars DataFrame 中单趟 (Single-Pass) 构建 2D C-Contiguous NumPy 矩阵块，
执行严格的 OHLC 字段校验与纯净数据表示，杜绝多重 Pivot 开销与伪数据填充。
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl

from cq.engine.utils.time_utils import extract_timestamps_series


# ============================================================================
# 1. 核心数据容器定义 (Data Containers)
# ============================================================================

class CustomFieldsDict(dict):
    """
    支持 keyword argument default=... 与属性访问的自定义特征字典
    """

    def get(self, key: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if key in self:
            return self[key]
        return default


class LazyCustomFields:
    """
    按需懒透视字典 (Lazy Custom Fields)
    首次访问字段时才利用 Polars 将其 pivot 成 (T, N) 的 2D C-Contiguous NumPy 矩阵并进行缓存。
    """

    def __init__(
        self,
        df: Optional[pl.DataFrame],
        all_timestamps: List[Any],
        all_symbols: List[str],
        initial_fields: Optional[Dict[str, np.ndarray]] = None,
    ):
        self._df = df
        self.all_timestamps = all_timestamps
        self.all_symbols = all_symbols
        self._cache: Dict[str, np.ndarray] = dict(initial_fields) if initial_fields else {}

    def __getitem__(self, key: str) -> np.ndarray:
        return self.get(key)

    def get(self, key: str, default: Optional[np.ndarray] = None) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]

        if self._df is not None and key in self._df.columns:
            time_col = "timestamp" if "timestamp" in self._df.columns else "datetime"
            pivoted = self._df.pivot(
                on="symbol",
                index=time_col,
                values=key,
                aggregate_function="first",
            ).sort(time_col)

            missing_syms = set(self.all_symbols) - set(pivoted.columns)
            for sym in missing_syms:
                pivoted = pivoted.with_columns(pl.lit(np.nan).alias(sym))

            matrix_df = pivoted.select(self.all_symbols)
            mat = np.ascontiguousarray(matrix_df.to_numpy(), dtype=np.float64)
            self._cache[key] = mat
            return mat

        if default is not None:
            return default
        raise KeyError(f"自定义列/特征 '{key}' 不存在。")

    def __contains__(self, key: str) -> bool:
        if key in self._cache:
            return True
        return self._df is not None and key in self._df.columns

    def keys(self) -> List[str]:
        keys_set = set(self._cache.keys())
        if self._df is not None:
            base_cols = {"symbol", "datetime", "timestamp", "open", "high", "low", "close", "volume", "amount", "back_adj_factor"}
            for col in self._df.columns:
                if col not in base_cols:
                    keys_set.add(col)
        return list(keys_set)


class AdjMarketData:
    """
    复权行情数据视角 (支持动态懒求值代理与显式矩阵)
    """

    def __init__(
        self,
        close: Optional[np.ndarray] = None,
        open_p: Optional[np.ndarray] = None,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        parent: Optional["MarketData"] = None,
    ):
        self._parent = parent
        self._explicit_close = close
        self._explicit_open = open_p
        self._explicit_high = high
        self._explicit_low = low

    @property
    def close(self) -> np.ndarray:
        if self._explicit_close is not None:
            return self._explicit_close
        if self._parent is not None:
            if self._parent.adj_factor is None:
                return self._parent.close
            return self._parent.close * self._parent.adj_factor
        raise AttributeError("AdjMarketData 既无显式矩阵也无关联 parent MarketData。")

    @property
    def open(self) -> np.ndarray:
        if self._explicit_open is not None:
            return self._explicit_open
        if self._parent is not None:
            if self._parent.adj_factor is None:
                return self._parent.open
            return self._parent.open * self._parent.adj_factor
        return self.close

    @property
    def high(self) -> np.ndarray:
        if self._explicit_high is not None:
            return self._explicit_high
        if self._parent is not None:
            if self._parent.adj_factor is None:
                return self._parent.high
            return self._parent.high * self._parent.adj_factor
        return self.close

    @property
    def low(self) -> np.ndarray:
        if self._explicit_low is not None:
            return self._explicit_low
        if self._parent is not None:
            if self._parent.adj_factor is None:
                return self._parent.low
            return self._parent.low * self._parent.adj_factor
        return self.close


class TimeSeriesTable:
    """
    通用时序特征/多因子表容器 (TimeSeriesTable)
    存储与 Master Clock (T, N) 对齐的多个 2D C-Contiguous 特征矩阵
    """

    def __init__(
        self,
        timestamps: np.ndarray,
        symbols: List[str],
        fields: Dict[str, np.ndarray],
    ):
        self.timestamps = np.ascontiguousarray(timestamps, dtype=np.int64)
        self.symbols = list(symbols)
        self.fields = {k: np.ascontiguousarray(v, dtype=np.float64) for k, v in fields.items()}
        first_mat = next(iter(self.fields.values())) if self.fields else np.empty((len(timestamps), len(symbols)))
        self.shape = first_mat.shape
        self.n_steps, self.n_symbols = self.shape

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self.fields:
            return self.fields[key]
        raise KeyError(f"TimeSeriesTable 中不存在特征列 '{key}'。可用字段: {list(self.fields.keys())}")

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def __getattr__(self, name: str) -> np.ndarray:
        if name in self.fields:
            return self.fields[name]
        raise AttributeError(f"TimeSeriesTable 不存在属性 '{name}'。可用字段: {list(self.fields.keys())}")

    def keys(self) -> List[str]:
        return list(self.fields.keys())

    def __repr__(self) -> str:
        return f"<TimeSeriesTable: T={self.n_steps}, N={self.n_symbols}, fields={list(self.fields.keys())}>"


class MarketData:
    """
    全市场行情矩阵容器 (MarketData)
    """

    def __init__(
        self,
        timestamps: np.ndarray,
        symbols: List[str],
        open_price: np.ndarray,
        high_price: np.ndarray,
        low_price: np.ndarray,
        close_price: np.ndarray,
        adj_close_price: Optional[np.ndarray] = None,
        adj_open_price: Optional[np.ndarray] = None,
        adj_high_price: Optional[np.ndarray] = None,
        adj_low_price: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None,
        amount: Optional[np.ndarray] = None,
        is_tradable: Optional[np.ndarray] = None,
        adj_factor: Optional[np.ndarray] = None,
        custom_fields: Optional[Union[Dict[str, np.ndarray], LazyCustomFields, CustomFieldsDict]] = None,
        tables: Optional[Dict[str, Any]] = None,
    ):
        self.timestamps = np.ascontiguousarray(timestamps)
        self.symbols = list(symbols)
        self.shape = open_price.shape  # (T, N)
        self.n_steps, self.n_symbols = self.shape

        # 1. 原始价格矩阵 (真实资金交割使用)
        self.open = np.ascontiguousarray(open_price, dtype=np.float64)
        self.high = np.ascontiguousarray(high_price, dtype=np.float64)
        self.low = np.ascontiguousarray(low_price, dtype=np.float64)
        self.close = np.ascontiguousarray(close_price, dtype=np.float64)

        # 2. 复权子视角与复权因子
        self.adj_factor = np.ascontiguousarray(adj_factor, dtype=np.float64) if adj_factor is not None else None

        if adj_close_price is not None:
            adj_c = np.ascontiguousarray(adj_close_price, dtype=np.float64)
            adj_o = np.ascontiguousarray(adj_open_price, dtype=np.float64) if adj_open_price is not None else self.open
            adj_h = np.ascontiguousarray(adj_high_price, dtype=np.float64) if adj_high_price is not None else self.high
            adj_l = np.ascontiguousarray(adj_low_price, dtype=np.float64) if adj_low_price is not None else self.low
            self.adj = AdjMarketData(close=adj_c, open_p=adj_o, high=adj_h, low=adj_l)
        else:
            self.adj = AdjMarketData(parent=self)

        # 3. 辅助量价矩阵 (保持纯净 None 表示)
        self.volume = np.ascontiguousarray(volume, dtype=np.float64) if volume is not None else None
        self.amount = np.ascontiguousarray(amount, dtype=np.float64) if amount is not None else None

        # 4. 可交易标志矩阵
        if is_tradable is not None:
            self.is_tradable = np.ascontiguousarray(is_tradable, dtype=np.bool_)
        else:
            if self.volume is None:
                self.is_tradable = np.ascontiguousarray(~np.isnan(self.close) & (self.close > 0), dtype=np.bool_)
            else:
                self.is_tradable = np.ascontiguousarray(~np.isnan(self.close) & (self.close > 0) & (self.volume > 0), dtype=np.bool_)

        # 5. 自定义特征与辅助表
        if custom_fields is not None:
            if isinstance(custom_fields, dict) and not isinstance(custom_fields, (CustomFieldsDict, LazyCustomFields)):
                self.custom_fields = CustomFieldsDict(custom_fields)
            else:
                self.custom_fields = custom_fields
        else:
            self.custom_fields = LazyCustomFields(df=None, all_timestamps=list(self.timestamps), all_symbols=self.symbols)

        self.tables: Dict[str, Any] = tables if tables is not None else {}

    def __getitem__(self, key: str) -> Any:
        return self.get_custom_field(key)

    def get_custom_field(self, key: str) -> Any:
        if isinstance(self.custom_fields, LazyCustomFields):
            return self.custom_fields[key]
        if isinstance(self.custom_fields, (dict, CustomFieldsDict)) and key in self.custom_fields:
            return self.custom_fields[key]
        if key in self.tables:
            return self.tables[key]
        raise KeyError(f"MarketData 中未找到自定义列或辅助表 '{key}'。")

    def __repr__(self) -> str:
        vol_repr = "None" if self.volume is None else f"shape={self.volume.shape}"
        return f"<MarketData: T={self.n_steps}, N={self.n_symbols}, symbols={len(self.symbols)}, volume={vol_repr}>"


# 别名兼容
MarketDataContainer = MarketData


class EventSnapshot:
    """
    单时间步离散/稀疏事件快照 (EventSnapshot)
    支持按标的代码 (str) 或标的索引 (int) 访问，未发生事件标的严格返回 None。
    """

    def __init__(self, events: Dict[str, Any], symbol_to_idx: Dict[str, int]):
        self._events = events  # {symbol: row_dict_or_data}
        self._symbol_to_idx = symbol_to_idx
        self._idx_to_sym: Dict[int, str] = {idx: sym for sym, idx in symbol_to_idx.items()}

    def get(self, key: Union[str, int], default: Any = None) -> Any:
        if isinstance(key, (int, np.integer)):
            sym = self._idx_to_sym.get(int(key))
            if sym is not None:
                return self._events.get(sym, default)
            return default
        return self._events.get(key, default)

    def __getitem__(self, key: Union[str, int]) -> Any:
        return self.get(key)

    def __contains__(self, key: Union[str, int]) -> bool:
        if isinstance(key, (int, np.integer)):
            sym = self._idx_to_sym.get(int(key))
            return sym in self._events if sym is not None else False
        return key in self._events

    def __len__(self) -> int:
        return len(self._events)

    def __bool__(self) -> bool:
        return len(self._events) > 0

    def keys(self) -> List[str]:
        return list(self._events.keys())

    def items(self):
        return self._events.items()

    def values(self):
        return self._events.values()

    def __repr__(self) -> str:
        return f"<EventSnapshot: {len(self._events)} events>"


class SparseEventContainer:
    """
    离散事件稀疏时序容器 (SparseEventContainer)
    按时间戳/时间步索引事件数据，绝不强行铺满 (T, N) 稠密矩阵。
    """

    def __init__(
        self,
        events_by_ts: Dict[int, Dict[str, Any]],
        all_timestamps: np.ndarray,
        all_symbols: List[str],
    ):
        self._events_by_ts = events_by_ts  # {ts_int64: {symbol: row_dict}}
        self._timestamps = all_timestamps
        self._symbols = all_symbols
        self._symbol_to_idx = {s: i for i, s in enumerate(all_symbols)}

    def get_snapshot(self, step: int, timestamp: Optional[int] = None) -> EventSnapshot:
        ts = int(self._timestamps[step]) if timestamp is None else int(timestamp)
        ev_dict = self._events_by_ts.get(ts, {})
        return EventSnapshot(events=ev_dict, symbol_to_idx=self._symbol_to_idx)

    def __repr__(self) -> str:
        total_events = sum(len(d) for d in self._events_by_ts.values())
        return f"<SparseEventContainer: {total_events} events over {len(self._events_by_ts)} timestamps>"


class StaticAttributeContainer:
    """
    静态属性映射容器 (StaticAttributeContainer)
    存储行业、概念板块等无时间轴静态映射，未分类标的查询返回 None。
    支持一对多列表聚合 (如概念列表)，并内置反向索引 (属性/概念 -> 标的代码列表)。
    """

    def __init__(self, mapping: Dict[str, Any], all_symbols: List[str]):
        self._mapping = mapping
        self._all_symbols = all_symbols
        self._symbol_to_idx = {s: i for i, s in enumerate(all_symbols)}
        self._reverse_mapping: Dict[str, List[str]] = {}
        self._build_reverse_index()

    def _build_reverse_index(self):
        """构建属性值到标的代码列表的反向哈希映射"""
        for sym, val in self._mapping.items():
            if isinstance(val, (str, int, float)):
                key_str = str(val)
                self._reverse_mapping.setdefault(key_str, []).append(sym)
            elif isinstance(val, (list, tuple, set)):
                for item in val:
                    if isinstance(item, (str, int, float)):
                        self._reverse_mapping.setdefault(str(item), []).append(sym)
                    elif isinstance(item, dict):
                        for sub_v in item.values():
                            if isinstance(sub_v, (str, int, float)):
                                self._reverse_mapping.setdefault(str(sub_v), []).append(sym)
            elif isinstance(val, dict):
                for sub_v in val.values():
                    if isinstance(sub_v, (str, int, float)):
                        self._reverse_mapping.setdefault(str(sub_v), []).append(sym)

    def get_symbols(self, attribute_val: Any) -> List[str]:
        """根据属性值（如行业名、概念板块名）反向查询属于该分类的全部标的代码列表"""
        return self._reverse_mapping.get(str(attribute_val), [])

    def get(self, key: Union[str, int], default: Any = None) -> Any:
        if isinstance(key, (int, np.integer)):
            if 0 <= key < len(self._all_symbols):
                sym = self._all_symbols[key]
                return self._mapping.get(sym, default)
            return default
        return self._mapping.get(key, default)

    def __getitem__(self, key: Union[str, int]) -> Any:
        return self.get(key)

    def __contains__(self, key: Union[str, int]) -> bool:
        if isinstance(key, (int, np.integer)):
            if 0 <= key < len(self._all_symbols):
                return self._all_symbols[key] in self._mapping
            return False
        return key in self._mapping

    def __len__(self) -> int:
        return len(self._mapping)

    def keys(self) -> List[str]:
        return list(self._mapping.keys())

    def items(self):
        return self._mapping.items()

    def values(self):
        return self._mapping.values()

    def __repr__(self) -> str:
        return f"<StaticAttributeContainer: {len(self._mapping)} symbols mapped, {len(self._reverse_mapping)} attributes>"


# 单点权威时间序列提取别名 (向后兼容)
parse_timestamps_series = extract_timestamps_series


# ============================================================================
# 2. 单趟极速对齐算子 (Alignment Engine)
# ============================================================================

def _align_df_to_grid(
    df_with_ts: pl.DataFrame,
    all_timestamps: Optional[np.ndarray],
    all_symbols: Optional[List[str]],
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    单趟极速坐标对齐算子：
    结合 Polars SIMD Hash Join 与 NumPy Int64 二分查找，计算 1D 连续展平索引 flat_idx 与 valid_mask。
    返回: (all_timestamps, all_symbols, flat_idx, valid_mask)
    """
    # 1. 确定时间轴
    if all_timestamps is None:
        all_timestamps = df_with_ts.select("_std_ts").unique().sort("_std_ts")["_std_ts"].to_numpy().astype(np.int64)
    else:
        all_timestamps = np.ascontiguousarray(all_timestamps, dtype=np.int64)

    # 2. 确定标的池
    if all_symbols is None:
        all_symbols = df_with_ts.select("symbol").unique().sort("symbol")["symbol"].to_list()
    else:
        df_sym_set = set(df_with_ts.select("symbol").unique()["symbol"].to_list())
        master_sym_set = set(all_symbols)
        if not df_sym_set.intersection(master_sym_set):
            all_symbols = sorted(list(df_sym_set))
        else:
            all_symbols = list(all_symbols)

    n_steps = len(all_timestamps)
    n_symbols = len(all_symbols)
    if n_steps == 0 or n_symbols == 0:
        raise ValueError("构建 2D 矩阵失败：时间步 (T) 或标的数 (N) 长度为 0。")

    # 3. 极速标的哈希映射 (Polars Hash Join) 与时间二分查找 (NumPy Int64 searchsorted)
    sym_map = pl.DataFrame({
        "symbol": all_symbols,
        "_s_idx": np.arange(n_symbols, dtype=np.int32),
    })
    df_mapped = df_with_ts.select(["symbol", "_std_ts"]).join(sym_map, on="symbol", how="left")

    s_idx = df_mapped["_s_idx"].fill_null(-1).to_numpy().astype(np.int64)
    df_ts = df_mapped["_std_ts"].to_numpy().astype(np.int64)
    t_idx = np.searchsorted(all_timestamps, df_ts)

    # 4. 计算有效掩码与 1D 展平物理索引 (flat_idx = t * N + s)
    valid_mask = (s_idx >= 0) & (t_idx < n_steps) & (all_timestamps[np.clip(t_idx, 0, n_steps - 1)] == df_ts)
    flat_idx = (t_idx[valid_mask] * n_symbols + s_idx[valid_mask]).astype(np.int64)

    return all_timestamps, all_symbols, flat_idx, valid_mask


def _fill_grid_column(
    series: pl.Series,
    n_steps: int,
    n_symbols: int,
    flat_idx: np.ndarray,
    valid_mask: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    单趟将 Polars Series 规整填充至预分配的 (T, N) C-Contiguous NumPy 矩阵 (利用 1D 连续内存直接赋值)
    """
    mat = np.full((n_steps, n_symbols), fill_value, dtype=np.float64)
    raw_vals = series.cast(pl.Float64).to_numpy()[valid_mask]
    mat.ravel()[flat_idx] = raw_vals
    return np.ascontiguousarray(mat, dtype=np.float64)


# ============================================================================
# 3. 工厂构建函数 (Factory Builders)
# ============================================================================

def build_market_data_from_df(
    df: pl.DataFrame,
    all_timestamps: Optional[np.ndarray] = None,
    all_symbols: Optional[List[str]] = None,
    is_main_clock: bool = True,
    columns: Optional[List[str]] = None,
) -> MarketData:
    """
    单趟极速构建主/副行情 MarketData 容器。
    """
    if df.is_empty():
        raise ValueError("无法从空 DataFrame 构建 MarketData。")

    if "symbol" not in df.columns:
        raise ValueError("数据源缺失必需的标的代码列：'symbol'")

    required_ohlc = ["open", "high", "low", "close"]
    missing_ohlc = [col for col in required_ohlc if col not in df.columns]
    if missing_ohlc:
        raise ValueError(f"数据源缺失必需的 OHLC 价格列：{missing_ohlc}，严禁伪造价格或自动推断。")

    # 1. 提取标准化时间戳
    ts_series = extract_timestamps_series(df)
    df_with_ts = df.with_columns(ts_series.alias("_std_ts"))

    # 2. 单趟坐标映射
    all_ts, all_syms, flat_idx, v_mask = _align_df_to_grid(df_with_ts, all_timestamps, all_symbols)
    n_steps, n_symbols = len(all_ts), len(all_syms)

    # 3. 单趟填充基础 OHLC 矩阵
    open_mat = _fill_grid_column(df_with_ts["open"], n_steps, n_symbols, flat_idx, v_mask)
    high_mat = _fill_grid_column(df_with_ts["high"], n_steps, n_symbols, flat_idx, v_mask)
    low_mat = _fill_grid_column(df_with_ts["low"], n_steps, n_symbols, flat_idx, v_mask)
    close_mat = _fill_grid_column(df_with_ts["close"], n_steps, n_symbols, flat_idx, v_mask)

    # 4. 成交量与成交额 (纯净表示：未提供则保持为 None)
    volume_mat = _fill_grid_column(df_with_ts["volume"], n_steps, n_symbols, flat_idx, v_mask) if "volume" in df_with_ts.columns else None
    amount_mat = _fill_grid_column(df_with_ts["amount"], n_steps, n_symbols, flat_idx, v_mask) if "amount" in df_with_ts.columns else None

    # 5. 复权因子
    adj_factor_mat: Optional[np.ndarray] = None
    if "back_adj_factor" in df_with_ts.columns:
        adj_factor_mat = _fill_grid_column(df_with_ts["back_adj_factor"], n_steps, n_symbols, flat_idx, v_mask, fill_value=1.0)
        adj_factor_mat = np.nan_to_num(adj_factor_mat, nan=1.0)

    # 6. 自定义特征列规整
    base_cols = {"symbol", "datetime", "timestamp", "_std_ts", "open", "high", "low", "close", "volume", "amount", "back_adj_factor"}
    requested_cols = set(columns or [])
    custom_fields_dict = CustomFieldsDict()

    for col in df_with_ts.columns:
        if col not in base_cols:
            if requested_cols and col not in requested_cols:
                continue
            if df_with_ts[col].dtype.is_numeric():
                custom_fields_dict[col] = _fill_grid_column(df_with_ts[col], n_steps, n_symbols, flat_idx, v_mask)

    return MarketData(
        timestamps=all_ts,
        symbols=all_syms,
        open_price=open_mat,
        high_price=high_mat,
        low_price=low_mat,
        close_price=close_mat,
        volume=volume_mat,
        amount=amount_mat,
        adj_factor=adj_factor_mat,
        custom_fields=custom_fields_dict,
    )


def build_timeseries_table_from_df(
    df: pl.DataFrame,
    all_timestamps: np.ndarray,
    all_symbols: List[str],
    columns: Optional[List[str]] = None,
) -> Union[MarketData, TimeSeriesTable]:
    """
    从 Polars DataFrame 构建时序特征表 (TimeSeriesTable) 或副行情表 (MarketData)
    """
    if df.is_empty():
        raise ValueError("无法从空 DataFrame 构建 TS 时序表。")

    if "symbol" not in df.columns:
        raise ValueError("TS 时序表必须包含 'symbol' 列。")

    # 1. 若完整包含 OHLC，直接构建副行情 MarketData
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        return build_market_data_from_df(
            df,
            all_timestamps=all_timestamps,
            all_symbols=all_symbols,
            is_main_clock=False,
            columns=columns,
        )

    # 2. 提取标准化时间戳并进行单趟坐标映射
    ts_series = extract_timestamps_series(df)
    df_with_ts = df.with_columns(ts_series.alias("_std_ts"))
    all_ts, all_syms, flat_idx, v_mask = _align_df_to_grid(df_with_ts, all_timestamps, all_symbols)
    n_steps, n_symbols = len(all_ts), len(all_syms)

    base_cols = {"symbol", "datetime", "timestamp", "_std_ts"}
    requested_cols = set(columns or [])
    fields_dict: Dict[str, np.ndarray] = {}

    for col in df_with_ts.columns:
        if col not in base_cols:
            if requested_cols and col not in requested_cols:
                continue
            if df_with_ts[col].dtype.is_numeric():
                fields_dict[col] = _fill_grid_column(df_with_ts[col], n_steps, n_symbols, flat_idx, v_mask)
            else:
                raise ValueError(
                    f"TS 时序表中的特征列 '{col}' 为非数值类型 ({df_with_ts[col].dtype})，无法构建连续数值矩阵。"
                    f"如为文本/描述等离散字段，请使用 event_table(df) 声明为离散事件表。"
                )

    if not fields_dict:
        raise ValueError(f"TS 时序表中未找到有效的数值特征列。可用列: {df_with_ts.columns}")

    return TimeSeriesTable(
        timestamps=all_ts,
        symbols=all_syms,
        fields=fields_dict,
    )


def build_event_container_from_df(
    df: pl.DataFrame,
    all_timestamps: np.ndarray,
    all_symbols: List[str],
) -> SparseEventContainer:
    """
    从 Polars DataFrame 构建按时间索引的离散事件稀疏容器 (SparseEventContainer)
    """
    if df.is_empty():
        return SparseEventContainer(events_by_ts={}, all_timestamps=all_timestamps, all_symbols=all_symbols)

    ts_series = extract_timestamps_series(df)
    df_with_ts = df.with_columns(ts_series.alias("_std_ts"))

    events_by_ts: Dict[int, Dict[str, Any]] = {}
    rows = df_with_ts.to_dicts()
    for row in rows:
        ts = int(row.pop("_std_ts"))
        sym = str(row.get("symbol", ""))
        if ts not in events_by_ts:
            events_by_ts[ts] = {}
        events_by_ts[ts][sym] = row

    return SparseEventContainer(
        events_by_ts=events_by_ts,
        all_timestamps=all_timestamps,
        all_symbols=all_symbols,
    )


def build_static_container_from_df(
    df: pl.DataFrame,
    all_symbols: List[str],
) -> StaticAttributeContainer:
    """
    从 Polars DataFrame 构建静态属性容器 (StaticAttributeContainer)
    支持一对多概念/行业平铺映射自动聚合为列表。
    """
    if "symbol" not in df.columns:
        raise ValueError("静态属性表必须包含 'symbol' 列。")

    value_cols = [c for c in df.columns if c != "symbol"]
    if not value_cols:
        raise ValueError("静态属性表除了 'symbol' 外必须至少包含 1 个属性列。")

    mapping: Dict[str, Any] = {}
    rows = df.to_dicts()

    for row in rows:
        sym = str(row["symbol"])
        if len(value_cols) == 1:
            val = row[value_cols[0]]
            if sym in mapping:
                existing = mapping[sym]
                if isinstance(existing, list):
                    existing.append(val)
                else:
                    mapping[sym] = [existing, val]
            else:
                mapping[sym] = val
        else:
            item = {k: v for k, v in row.items() if k != "symbol"}
            if sym in mapping:
                existing = mapping[sym]
                if isinstance(existing, list):
                    existing.append(item)
                else:
                    mapping[sym] = [existing, item]
            else:
                mapping[sym] = item

    return StaticAttributeContainer(mapping=mapping, all_symbols=all_symbols)
