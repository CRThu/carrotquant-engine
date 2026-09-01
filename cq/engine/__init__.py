"""
CarrotQuant Engine (cq.engine)

高性能全市场 Numba 事件驱动量化回测与撮合引擎。
"""

__version__ = "1.2.0"

from cq.engine.runner import Engine
from cq.engine.matching import MatchingMode
from cq.engine.strategy.base import strategy
from cq.engine.strategy.context import BarContext, TableContext
from cq.engine.feed.column_loader import ColumnDataLoader, MarketData, MarketDataContainer
from cq.engine.feed.matrix_builder import TimeSeriesTable, SparseEventContainer, StaticAttributeContainer, EventSnapshot
from cq.engine.feed.feed_loader import load_feed, stream_feed, ts_table, event_table, static_table, TableSpec
from cq.engine.feed.chunk_streamer import ChunkStreamer
from cq.engine.indicators.dynamic_ma import BaseDynamicIndicator, calc_sma_step_jit, calc_ema_step_jit
from cq.engine.analytics.post_process import BacktestResult

__all__ = [
    "Engine",
    "MatchingMode",
    "strategy",
    "BarContext",
    "TableContext",
    "ColumnDataLoader",
    "MarketData",
    "MarketDataContainer",
    "TimeSeriesTable",
    "SparseEventContainer",
    "StaticAttributeContainer",
    "EventSnapshot",
    "TableSpec",
    "ts_table",
    "event_table",
    "static_table",
    "ChunkStreamer",
    "load_feed",
    "stream_feed",
    "BaseDynamicIndicator",
    "calc_sma_step_jit",
    "calc_ema_step_jit",
    "BacktestResult",
    "__version__",
]
