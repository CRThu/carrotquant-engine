"""
SoA (Structure of Arrays) 预分配状态容器

在内存中一次性预分配持仓、资金、资产历史与交易日志大数组，
回测运行期间完全零动态内存分配 (Zero Allocation)，彻底消除 GC 与堆分配开销。
"""

import numpy as np


class EngineState:
    """
    回测引擎运行期 SoA 状态对象
    """

    def __init__(
        self,
        n_steps: int,
        n_symbols: int,
        initial_cash: float = 1_000_000.0,
        max_trades: int = 1_000_000,
    ):
        self.n_steps = n_steps
        self.n_symbols = n_symbols
        self.initial_cash = initial_cash

        # 1. 账户资产状态
        self.cash = float(initial_cash)
        self.positions = np.zeros(n_symbols, dtype=np.float64)       # 持仓数量 (N,)
        self.avg_costs = np.zeros(n_symbols, dtype=np.float64)        # 持仓成本价 (N,)
        self.portfolio_value = np.zeros(n_steps, dtype=np.float64)  # 逐 Bar 账户总资产 (T,)
        self.cash_history = np.zeros(n_steps, dtype=np.float64)     # 逐 Bar 现金历史 (T,)

        # 2. 预分配交易日志大数组 [Max_Trades, 7]
        # 列定义: [step_idx, symbol_idx, side (1=Buy, -1=Sell), amount, price, fee, cash_after]
        self.max_trades = max_trades
        self.trade_logs = np.zeros((max_trades, 7), dtype=np.float64)
        self.trade_count = np.array([0], dtype=np.int64)  # 使用 1D numpy array 存储计数器以方便 Numba JIT 引用传参


    def reset(self):
        """重置状态"""
        self.cash = float(self.initial_cash)
        self.positions.fill(0.0)
        self.avg_costs.fill(0.0)
        self.portfolio_value.fill(0.0)
        self.cash_history.fill(0.0)
        self.trade_logs.fill(0.0)
        self.trade_count[0] = 0

