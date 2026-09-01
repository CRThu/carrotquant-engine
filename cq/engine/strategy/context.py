"""
BarContext: 策略运行期上下文与物理防未来函数切片断言

提供当前 Bar (时间步 t) 的全市场行情快照切片、账户资金/持仓状态以及下单接口。
严格限制时间切片边界在 [:t+1]，任何读取 t+1 以上数据的行为在物理内存层面抛出 IndexError。
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
import numpy as np

from cq.engine.utils.time_utils import ts_to_iso_str
from cq.engine.feed.matrix_builder import (
    EventSnapshot,
    SparseEventContainer,
    StaticAttributeContainer,
    TimeSeriesTable,
    MarketData,
)


class TableContext:
    """
    辅助表/从属行情/多因子时序表运行期快照上下文 (TableContext)
    以命名空间形式封装辅助时序表在时间步 t 的切片与防未来历史。
    支持统一语法：
    - 当前快照向量 (N,): ctx.get("index").close 或 ctx.get("valuation").pe_ttm
    - 历史矩阵切片 (t+1, N): ctx.get("index").close_history 或 ctx.get("valuation").pe_ttm_history
    - 动态列提取: ctx.get("valuation")["pe_ttm"] 或 ctx.get("valuation").get_history("pe_ttm")
    """

    def __init__(self, table: Any, step: int):
        self._table = table
        self._step = step

    @property
    def open(self) -> np.ndarray:
        if hasattr(self._table, "open"):
            return self._table.open[self._step, :]
        return self.__getattr__("open")

    @property
    def high(self) -> np.ndarray:
        if hasattr(self._table, "high"):
            return self._table.high[self._step, :]
        return self.__getattr__("high")

    @property
    def low(self) -> np.ndarray:
        if hasattr(self._table, "low"):
            return self._table.low[self._step, :]
        return self.__getattr__("low")

    @property
    def close(self) -> np.ndarray:
        if hasattr(self._table, "close"):
            return self._table.close[self._step, :]
        return self.__getattr__("close")

    @property
    def volume(self) -> Optional[np.ndarray]:
        if hasattr(self._table, "volume") and self._table.volume is not None:
            return self._table.volume[self._step, :]
        return None

    @property
    def amount(self) -> Optional[np.ndarray]:
        if hasattr(self._table, "amount") and self._table.amount is not None:
            return self._table.amount[self._step, :]
        return None

    @property
    def is_tradable(self) -> np.ndarray:
        if hasattr(self._table, "is_tradable"):
            return self._table.is_tradable[self._step, :]
        return np.ones(self._table.n_symbols, dtype=bool)

    @property
    def price(self) -> float:
        """单标的/宏观基准快捷价格 (优先返回有效数值)"""
        c_row = self.close
        valid_vals = c_row[~np.isnan(c_row)]
        if len(valid_vals) > 0:
            return float(valid_vals[0])
        return float(c_row[0])

    @property
    def open_history(self) -> np.ndarray:
        if hasattr(self._table, "open"):
            return self._table.open[: self._step + 1, :]
        return self.get_history("open")

    @property
    def high_history(self) -> np.ndarray:
        if hasattr(self._table, "high"):
            return self._table.high[: self._step + 1, :]
        return self.get_history("high")

    @property
    def low_history(self) -> np.ndarray:
        if hasattr(self._table, "low"):
            return self._table.low[: self._step + 1, :]
        return self.get_history("low")

    @property
    def close_history(self) -> np.ndarray:
        if hasattr(self._table, "close"):
            return self._table.close[: self._step + 1, :]
        return self.get_history("close")

    def get(self, key: str) -> np.ndarray:
        """获取当前时间步 t 的特征向量 (N,)"""
        if isinstance(self._table, TimeSeriesTable) and key in self._table.fields:
            return self._table.fields[key][self._step, :]
        if hasattr(self._table, "custom_fields") and key in self._table.custom_fields:
            return self._table.custom_fields[key][self._step, :]
        if hasattr(self._table, key):
            val = getattr(self._table, key)
            if isinstance(val, np.ndarray) and val.ndim == 2:
                return val[self._step, :]
        raise KeyError(f"TableContext 中未找到特征列/自定义字段 '{key}'。可用列: {self.columns}")

    def get_history(self, key: str) -> np.ndarray:
        """获取截至当前时间步 t 的特征历史矩阵切片 [:t+1, :] (严格防未来)"""
        if isinstance(self._table, TimeSeriesTable) and key in self._table.fields:
            return self._table.fields[key][: self._step + 1, :]
        if hasattr(self._table, "custom_fields") and key in self._table.custom_fields:
            return self._table.custom_fields[key][: self._step + 1, :]
        if hasattr(self._table, key):
            val = getattr(self._table, key)
            if isinstance(val, np.ndarray) and val.ndim == 2:
                return val[: self._step + 1, :]
        raise KeyError(f"TableContext 中未找到特征列/自定义字段 '{key}' 的历史切片。可用列: {self.columns}")

    def __getitem__(self, key: str) -> np.ndarray:
        return self.get(key)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"TableContext 没有属性 '{name}'。")
        if name.endswith("_history"):
            base_name = name[:-8]
            return self.get_history(base_name)
        return self.get(name)

    @property
    def columns(self) -> List[str]:
        """该时序表包含的所有可用特征列名"""
        if isinstance(self._table, TimeSeriesTable):
            return list(self._table.fields.keys())
        if isinstance(self._table, MarketData):
            cols = ["open", "high", "low", "close"]
            if self._table.volume is not None:
                cols.append("volume")
            if self._table.amount is not None:
                cols.append("amount")
            if hasattr(self._table.custom_fields, "keys"):
                cols.extend(list(self._table.custom_fields.keys()))
            return cols
        return []

    def __repr__(self) -> str:
        return f"<TableContext: step={self._step}, columns={self.columns}>"


class AdjContext:
    """
    策略运行期复权行情快照与历史切片视角 (动态懒算与后向兼容)
    """

    def __init__(self, ctx: "BarContext"):
        self._ctx = ctx

    @property
    def open(self) -> np.ndarray:
        if self._ctx._has_explicit_adj:
            return self._ctx._adj_open_mat[self._ctx.step, :]
        if self._ctx._adj_factor is None:
            return self._ctx.open
        return self._ctx.open * self._ctx._adj_factor[self._ctx.step, :]

    @property
    def high(self) -> np.ndarray:
        if self._ctx._has_explicit_adj:
            return self._ctx._adj_high_mat[self._ctx.step, :]
        if self._ctx._adj_factor is None:
            return self._ctx.high
        return self._ctx.high * self._ctx._adj_factor[self._ctx.step, :]

    @property
    def low(self) -> np.ndarray:
        if self._ctx._has_explicit_adj:
            return self._ctx._adj_low_mat[self._ctx.step, :]
        if self._ctx._adj_factor is None:
            return self._ctx.low
        return self._ctx.low * self._ctx._adj_factor[self._ctx.step, :]

    @property
    def close(self) -> np.ndarray:
        if self._ctx._has_explicit_adj:
            return self._ctx._adj_close_mat[self._ctx.step, :]
        if self._ctx._adj_factor is None:
            return self._ctx.close
        return self._ctx.close * self._ctx._adj_factor[self._ctx.step, :]

    @property
    def open_history(self) -> np.ndarray:
        if self._ctx._has_explicit_adj:
            return self._ctx._adj_open_mat[: self._ctx.step + 1, :]
        if self._ctx._adj_factor is None:
            return self._ctx.open_history
        return self._ctx.open_history * self._ctx._adj_factor[: self._ctx.step + 1, :]

    @property
    def high_history(self) -> np.ndarray:
        if self._ctx._has_explicit_adj:
            return self._ctx._adj_high_mat[: self._ctx.step + 1, :]
        if self._ctx._adj_factor is None:
            return self._ctx.high_history
        return self._ctx.high_history * self._ctx._adj_factor[: self._ctx.step + 1, :]

    @property
    def low_history(self) -> np.ndarray:
        if self._ctx._has_explicit_adj:
            return self._ctx._adj_low_mat[: self._ctx.step + 1, :]
        if self._ctx._adj_factor is None:
            return self._ctx.low_history
        return self._ctx.low_history * self._ctx._adj_factor[: self._ctx.step + 1, :]

    @property
    def close_history(self) -> np.ndarray:
        if self._ctx._has_explicit_adj:
            return self._ctx._adj_close_mat[: self._ctx.step + 1, :]
        if self._ctx._adj_factor is None:
            return self._ctx.close_history
        return self._ctx.close_history * self._ctx._adj_factor[: self._ctx.step + 1, :]


class BarContext:
    """
    策略交互上下文 (BarContext)
    """

    def __init__(
        self,
        step: int,
        n_symbols: int,
        timestamps: np.ndarray,
        open_mat: np.ndarray,
        high_mat: np.ndarray,
        low_mat: np.ndarray,
        close_mat: np.ndarray,
        adj_close_mat: Optional[np.ndarray] = None,
        adj_open_mat: Optional[np.ndarray] = None,
        adj_high_mat: Optional[np.ndarray] = None,
        adj_low_mat: Optional[np.ndarray] = None,
        volume_mat: Optional[np.ndarray] = None,
        amount_mat: Optional[np.ndarray] = None,
        is_tradable_mat: Optional[np.ndarray] = None,
        adj_factor: Optional[np.ndarray] = None,
        custom_fields: Optional[Any] = None,
        tables: Optional[Dict[str, Any]] = None,
        positions: Optional[np.ndarray] = None,
        cash: float = 0.0,
        orders_buffer: Optional[List] = None,
        warmup_steps: int = 0,
    ):
        self.step = step
        self.n_symbols = n_symbols
        self._timestamps = timestamps
        self.warmup_steps = warmup_steps

        self._open_mat = open_mat
        self._high_mat = high_mat
        self._low_mat = low_mat
        self._close_mat = close_mat

        self._adj_factor = adj_factor
        self._custom_fields = custom_fields if custom_fields is not None else {}
        self._tables: Dict[str, Any] = tables if tables is not None else {}
        self._has_explicit_adj = adj_close_mat is not None

        self._adj_close_mat = adj_close_mat if adj_close_mat is not None else close_mat
        self._adj_open_mat = adj_open_mat if adj_open_mat is not None else open_mat
        self._adj_high_mat = adj_high_mat if adj_high_mat is not None else high_mat
        self._adj_low_mat = adj_low_mat if adj_low_mat is not None else low_mat

        self._volume_mat = volume_mat
        self._amount_mat = amount_mat
        self._is_tradable_mat = is_tradable_mat

        # 当前时间步 t 的 1D 快照切片 (N,)
        self.open = open_mat[step, :]
        self.high = high_mat[step, :]
        self.low = low_mat[step, :]
        self.close = close_mat[step, :]

        self.volume: Optional[np.ndarray] = volume_mat[step, :] if volume_mat is not None else None
        self.amount: Optional[np.ndarray] = amount_mat[step, :] if amount_mat is not None else None
        self.is_tradable: np.ndarray = (
            is_tradable_mat[step, :] if is_tradable_mat is not None else np.ones(n_symbols, dtype=bool)
        )

        # 账户状态
        self.positions = positions
        self.cash = cash

        # 下单指令缓冲 [(order_id, order_type, side, symbol_idx, amount, price), ...]
        self.orders_buffer = orders_buffer if orders_buffer is not None else []
        self._order_counter = 0
        self.canceled_order_ids = set()

        # 复权子视角
        self.adj = AdjContext(self)

    @property
    def is_warmup(self) -> bool:
        """是否处于预热期"""
        return self.step < self.warmup_steps

    def update_step(self, step: int, cash: float):
        """在主循环中原地更新时间步与指针"""
        self.step = step
        self.cash = cash
        self.orders_buffer.clear()
        self.canceled_order_ids.clear()

        self.open = self._open_mat[step, :]
        self.high = self._high_mat[step, :]
        self.low = self._low_mat[step, :]
        self.close = self._close_mat[step, :]
        if self._volume_mat is not None:
            self.volume = self._volume_mat[step, :]
        if self._amount_mat is not None:
            self.amount = self._amount_mat[step, :]
        if self._is_tradable_mat is not None:
            self.is_tradable = self._is_tradable_mat[step, :]

    def get(self, name: str) -> Any:
        """
        获取当前时间步 t 的辅助表、事件快照或自定义特征切片。
        1. 辅助行情/多因子时序表：返回 TableContext(表对象, step)
        2. 离散事件表：返回 EventSnapshot(当日事件字典)
        3. 静态属性表：返回 StaticAttributeContainer
        4. 自定义列矩阵：返回当前时间步 1D 快照切片 (N,)
        """
        if name in self._tables:
            tbl = self._tables[name]
            if isinstance(tbl, (MarketData, TimeSeriesTable)) or (hasattr(tbl, "open") and hasattr(tbl, "close")):
                return TableContext(tbl, self.step)
            elif isinstance(tbl, SparseEventContainer):
                return tbl.get_snapshot(self.step, self.timestamp)
            elif isinstance(tbl, StaticAttributeContainer):
                return tbl
            elif isinstance(tbl, np.ndarray) and tbl.ndim == 2:
                return tbl[self.step, :]
            return tbl

        if name in self._custom_fields:
            mat = self._custom_fields[name]
            return mat[self.step, :]

        raise KeyError(f"BarContext 中未找到自定义列或辅助表 '{name}'。")

    def get_history(self, name: str) -> np.ndarray:
        """获取截至当前时间步 t 的自定义列历史 2D 切片 [:t+1, :] (物理严格防未来)"""
        if name in self._tables:
            tbl = self._tables[name]
            if isinstance(tbl, (MarketData, TimeSeriesTable)) or hasattr(tbl, "close"):
                if hasattr(tbl, "close"):
                    return tbl.close[: self.step + 1, :]
            elif isinstance(tbl, np.ndarray) and tbl.ndim == 2:
                return tbl[: self.step + 1, :]

        if name in self._custom_fields:
            mat = self._custom_fields[name]
            return mat[: self.step + 1, :]

        raise KeyError(f"BarContext 中未找到自定义列或辅助表 '{name}' 的历史切片。")

    def __getitem__(self, name: str) -> Any:
        """字典下标简写支持 `ctx['index']` 或 `ctx['factor']`"""
        return self.get(name)

    @property
    def custom(self):
        """自定义特征映射代理"""
        return self._custom_fields

    @property
    def timestamp(self) -> int:
        """当前 Bar 的 UTC 毫秒时间戳 (Int64)"""
        val = self._timestamps[self.step]
        if isinstance(val, (int, np.integer)):
            return int(val)
        return int(val)

    @property
    def datetime(self) -> str:
        """当前 Bar 时间戳 ISO 格式字符串 (按需懒转换)"""
        val = self._timestamps[self.step]
        return ts_to_iso_str(val)

    # 单标的便捷属性
    @property
    def price(self) -> float:
        """单标的快捷当前收盘价"""
        return float(self.close[0])

    def buy(self, symbol_idx: int = 0, amount: float = 0.0) -> int:
        """挂买入市价单 (pos += amount)"""
        if amount > 0:
            self.orders_buffer.append((1, symbol_idx, float(amount)))
            return 0
        return 0

    def sell(self, symbol_idx: int = 0, amount: float = 0.0) -> int:
        """挂卖出市价单 (pos -= amount，天然支持做空)"""
        if amount > 0:
            self.orders_buffer.append((-1, symbol_idx, float(amount)))
            return 0
        return 0

    def buy_limit(self, symbol_idx: int = 0, amount: float = 0.0, price: float = 0.0) -> int:
        """挂买入限价单 (当市场最低价 <= limit_price 时触发买入)"""
        if amount > 0 and price > 0:
            self._order_counter += 1
            order_id = self._order_counter
            self.orders_buffer.append((order_id, 1, 1, symbol_idx, float(amount), float(price)))
            return order_id
        return 0

    def sell_limit(self, symbol_idx: int = 0, amount: float = 0.0, price: float = 0.0) -> int:
        """挂卖无限价单 (当市场最高价 >= limit_price 时触发卖出)"""
        if amount > 0 and price > 0:
            self._order_counter += 1
            order_id = self._order_counter
            self.orders_buffer.append((order_id, 1, -1, symbol_idx, float(amount), float(price)))
            return order_id
        return 0

    def cancel_order(self, order_id: int):
        """撤销指定 ID 的未成交订单"""
        if order_id > 0:
            self.canceled_order_ids.add(order_id)

    def buy_single(self, amount: float):
        """单标的快捷买入"""
        self.buy(0, amount)

    def sell_single(self, amount: float):
        """单标的快捷卖出"""
        self.sell(0, amount)

    # 物理边界切片限制 [:step+1] 严格防未来函数
    @property
    def open_history(self) -> np.ndarray:
        return self._open_mat[: self.step + 1, :]

    @property
    def close_history(self) -> np.ndarray:
        return self._close_mat[: self.step + 1, :]

    @property
    def high_history(self) -> np.ndarray:
        return self._high_mat[: self.step + 1, :]

    @property
    def low_history(self) -> np.ndarray:
        return self._low_mat[: self.step + 1, :]
