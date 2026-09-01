"""
按时间 Window 流式 Chunk 分块器 (ChunkStreamer)

用于超大规模全市场行情数据（如 1m 频段多年数据）的流式推流与 Window 分块加载。
"""

from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
from cq.engine.feed.matrix_builder import MarketDataContainer, MarketData, CustomFieldsDict


class ChunkStreamer:
    """
    行情 Chunk 流式分块推流器
    """

    def __init__(self, data: MarketData, chunk_size: int = 1000):
        """
        初始化 ChunkStreamer

        Args:
            data: 完整的 MarketData 实例
            chunk_size: 每个 Chunk 包含的时间步 Bar 数量
        """
        self.data = data
        self.chunk_size = chunk_size
        self.total_steps = data.n_steps

    def iter_chunks(self) -> Generator[Tuple[int, int, MarketData], None, None]:
        """
        按 chunk_size 产生时间片段 [start_idx, end_idx) 的 MarketData 切片
        """
        for start_idx in range(0, self.total_steps, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, self.total_steps)

            vol_chunk = self.data.volume[start_idx:end_idx] if self.data.volume is not None else None
            amt_chunk = self.data.amount[start_idx:end_idx] if self.data.amount is not None else None
            adj_f_chunk = self.data.adj_factor[start_idx:end_idx] if self.data.adj_factor is not None else None
            tradable_chunk = self.data.is_tradable[start_idx:end_idx] if self.data.is_tradable is not None else None

            # 切片自定义字段
            custom_chunk: Optional[CustomFieldsDict] = None
            if hasattr(self.data, "custom_fields") and self.data.custom_fields:
                custom_chunk = CustomFieldsDict()
                for k in self.data.custom_fields.keys():
                    try:
                        mat = self.data.custom_fields[k]
                        if isinstance(mat, np.ndarray) and mat.ndim == 2:
                            custom_chunk[k] = mat[start_idx:end_idx]
                    except Exception:
                        pass

            chunk_container = MarketData(
                timestamps=self.data.timestamps[start_idx:end_idx],
                symbols=self.data.symbols,
                open_price=self.data.open[start_idx:end_idx],
                high_price=self.data.high[start_idx:end_idx],
                low_price=self.data.low[start_idx:end_idx],
                close_price=self.data.close[start_idx:end_idx],
                volume=vol_chunk,
                amount=amt_chunk,
                is_tradable=tradable_chunk,
                adj_factor=adj_f_chunk,
                custom_fields=custom_chunk,
                tables=self.data.tables,
            )
            yield start_idx, end_idx, chunk_container
