"""
Engine: 回测引擎主入口与 Bar 循环调度器

连接行情数据、SoA 状态数组、撮合核算子与策略逻辑。
提供统一的 Stream-Native 回测调度入口 engine.run()，支持统一 Dict 传参与 Duck Typing 数据协议。
"""

from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union
import numpy as np
from numba import njit

from cq.engine.feed.column_loader import MarketData
from cq.engine.feed.feed_loader import load_feed, stream_feed
from cq.engine.state import EngineState
from cq.engine.matching import (
    execute_trade_jit,
    get_execution_price,
    check_limit_order_triggered_jit,
    MatchingMode,
    MATCHING_MODE_CLOSE,
    MATCHING_MODE_OPEN,
)
from cq.engine.strategy.context import BarContext
from cq.engine.analytics.post_process import BacktestResult


@njit(fastmath=True, nogil=True)
def run_engine_jit_kernel(
    open_mat: np.ndarray,
    high_mat: np.ndarray,
    low_mat: np.ndarray,
    close_mat: np.ndarray,
    volume_mat: np.ndarray,
    amount_mat: np.ndarray,
    is_tradable_mat: np.ndarray,
    signals_mat: np.ndarray,  # 预生成的信号矩阵 (T, N) : 1=Buy, -1=Sell, 0=Hold
    amounts_mat: np.ndarray,  # 下单数量矩阵 (T, N)
    matching_mode: int,
    fee_rate: float,
    min_fee: float,
    stamp_duty: float,
    slippage: float,
    positions: np.ndarray,
    avg_costs: np.ndarray,
    cash_arr: np.ndarray,
    portfolio_value: np.ndarray,
    cash_history: np.ndarray,
    trade_logs: np.ndarray,
    trade_count: np.ndarray,
    max_volume_ratio: float = 1.0,
    adj_open_mat: np.ndarray = None,
    adj_close_mat: np.ndarray = None,
    long_margin_ratio: float = 1.0,
    short_margin_ratio: float = 1.0,
    warmup_steps: int = 0,
    lot_size: int = 1,
    global_step_offset: int = 0,
):
    """
    全 JIT 内核化主 Bar 循环 (Full JIT Loop Kernel)
    """
    n_steps, n_symbols = open_mat.shape
    adj_open = adj_open_mat if adj_open_mat is not None else open_mat
    adj_close = adj_close_mat if adj_close_mat is not None else close_mat

    for t in range(n_steps):
        is_warmup = (t < warmup_steps)
        global_step = global_step_offset + t

        # 1. 扫描与撮合 (预热期不撮合)
        if not is_warmup:
            if matching_mode == MATCHING_MODE_OPEN:
                # OPEN 撮合模式：t 步在开盘按 open_mat[t, i] 撮合 t-1 步产生的信号
                if t > 0 and (t - 1) >= warmup_steps:
                    for i in range(n_symbols):
                        sig = signals_mat[t - 1, i]
                        if sig != 0 and is_tradable_mat[t, i]:
                            amt = amounts_mat[t - 1, i]
                            if amt > 0:
                                raw_p = open_mat[t, i]
                                adj_p = adj_open[t, i]
                                vol_val = volume_mat[t, i] if volume_mat is not None else 0.0

                                execute_trade_jit(
                                    step_idx=global_step,
                                    symbol_idx=i,
                                    side=int(sig),
                                    target_amount=amt,
                                    raw_price=raw_p,
                                    adj_price=adj_p,
                                    fee_rate=fee_rate,
                                    min_fee=min_fee,
                                    stamp_duty=stamp_duty,
                                    slippage=slippage,
                                    positions=positions,
                                    avg_costs=avg_costs,
                                    cash_arr=cash_arr,
                                    trade_logs=trade_logs,
                                    trade_count=trade_count,
                                    volume=vol_val,
                                    max_volume_ratio=max_volume_ratio,
                                    long_margin_ratio=long_margin_ratio,
                                    short_margin_ratio=short_margin_ratio,
                                    lot_size=lot_size,
                                )
            else:
                # 当 Bar 撮合模式 (CLOSE / VWAP / TWAP)
                for i in range(n_symbols):
                    sig = signals_mat[t, i]
                    if sig != 0 and is_tradable_mat[t, i]:
                        amt = amounts_mat[t, i]
                        if amt > 0:
                            raw_p = get_execution_price(
                                matching_mode,
                                open_mat[t, i],
                                high_mat[t, i],
                                low_mat[t, i],
                                close_mat[t, i],
                                volume_mat[t, i],
                                amount_mat[t, i],
                            )
                            adj_p = get_execution_price(
                                matching_mode,
                                adj_open[t, i],
                                high_mat[t, i],
                                low_mat[t, i],
                                adj_close[t, i],
                                volume_mat[t, i],
                                amount_mat[t, i],
                            )

                            vol_val = volume_mat[t, i] if volume_mat is not None else 0.0

                            execute_trade_jit(
                                step_idx=global_step,
                                symbol_idx=i,
                                side=int(sig),
                                target_amount=amt,
                                raw_price=raw_p,
                                adj_price=adj_p,
                                fee_rate=fee_rate,
                                min_fee=min_fee,
                                stamp_duty=stamp_duty,
                                slippage=slippage,
                                positions=positions,
                                avg_costs=avg_costs,
                                cash_arr=cash_arr,
                                trade_logs=trade_logs,
                                trade_count=trade_count,
                                volume=vol_val,
                                max_volume_ratio=max_volume_ratio,
                                long_margin_ratio=long_margin_ratio,
                                short_margin_ratio=short_margin_ratio,
                                lot_size=lot_size,
                            )

        # 2. 极速计算当前 Bar 的账户净资产 (PV = Cash + sum(pos * close)，带 NaN 停牌保护)
        current_cash = cash_arr[0]
        cash_history[t] = current_cash

        pos_val = 0.0
        for i in range(n_symbols):
            if positions[i] != 0.0:
                c_p = close_mat[t, i]
                if not np.isnan(c_p) and c_p > 0.0:
                    pos_val += positions[i] * c_p
                elif not np.isnan(open_mat[t, i]) and open_mat[t, i] > 0.0:
                    pos_val += positions[i] * open_mat[t, i]
                else:
                    pos_val += positions[i] * avg_costs[i]

        portfolio_value[t] = current_cash + pos_val


class Engine:
    """
    CarrotQuant 通用事件驱动与向量化回测引擎主控制器
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        fee_rate: float = 0.0003,      # 佣金率
        min_fee: float = 5.0,          # 最小佣金 5 元
        stamp_duty: float = 0.0005,    # 卖出印花税
        slippage: float = 0.0001,      # 交易滑点
        max_volume_ratio: float = 1.0, # 盘口最大成交量比例 (如 0.1 表示单笔最多成交当前 Bar 10% 流动性)
        matching_mode: Union[int, str, MatchingMode] = MATCHING_MODE_CLOSE,  # 支持字符串或 Enum
        max_trades: int = 1_000_000,
        long_margin_ratio: float = 1.0,   # 做多保证金率，默认 1.0 (100% 现货无杠杆)
        short_margin_ratio: float = 1.0,  # 做空保证金率，默认 1.0 (100% 保证金)
        margin_interest_rate: float = 0.0, # 做多融资年化利率 (如 0.06 表示年化 6%)
        borrow_interest_rate: float = 0.0, # 做空融券年化利率 (如 0.08 表示年化 8%)
        enable_t1: bool = False,           # 是否启用 A 股 T+1 交易规则 (当日买入次日可卖)
        lot_size: int = 1,                 # 交易手大小 (A 股买入通常为 100 股整倍数)
        annualization_factor: Optional[int] = None, # 自定义年化因子 (默认自动推断)
    ):
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.min_fee = min_fee
        self.stamp_duty = stamp_duty
        self.slippage = slippage
        self.max_volume_ratio = max_volume_ratio
        self.matching_mode = MatchingMode.parse(matching_mode)
        self.max_trades = max_trades
        self.long_margin_ratio = long_margin_ratio
        self.short_margin_ratio = short_margin_ratio
        self.margin_interest_rate = margin_interest_rate
        self.borrow_interest_rate = borrow_interest_rate
        self.enable_t1 = enable_t1
        self.lot_size = lot_size
        self.annualization_factor = annualization_factor

    def run(
        self,
        strategy: Optional[Callable[[BarContext], None]] = None,
        signals: Optional[np.ndarray] = None,
        amounts: Optional[np.ndarray] = None,
        data: Union[dict, MarketData, Iterable[MarketData], Any] = None,
        warmup_steps: int = 0,
    ) -> BacktestResult:
        """
        统一回测运行入口 (Unified Stream-Native Engine Run API)

        自动支持:
          1. Python 回调策略: engine.run(strategy=my_strat, data=data)
          2. Fast Vectorized 模式: engine.run(signals=signals, amounts=amounts, data=data)
          3. 统一 Dict / Duck Typing 数据源: engine.run(strategy=my_strat, data={"stock": df, "index": index_df})
          4. 磁盘分块流式模式: engine.run(strategy=my_strat, data=scan_chunks(...), warmup_steps=10)
        """
        if data is None:
            raise ValueError("Must provide data or data stream generator to engine.run()")

        # 归一化为 Chunks 流
        if isinstance(data, MarketData):
            chunk_stream = [data]
        elif hasattr(data, "iter_chunks") or (isinstance(data, (Iterable, Iterator, Generator)) and not isinstance(data, (dict, list, tuple, str))):
            chunk_stream = stream_feed(data)
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], MarketData):
            chunk_stream = data
        else:
            # 标准字典、DataFrame 或 Duck Typing 对象
            loaded_data = load_feed(data)
            chunk_stream = [loaded_data]

        all_portfolio_values = []
        all_cash_histories = []
        all_timestamps = []
        all_trade_logs = []
        all_symbols_seen = []

        current_cash = self.initial_cash
        global_step_offset = 0

        # 基于 Symbol 字典的跨 Chunk 无损状态桥接 (彻底杜绝 N1 != N2 广播崩溃与错位)
        positions_map: Dict[str, float] = {}
        avg_costs_map: Dict[str, float] = {}
        available_positions_map: Dict[str, float] = {}
        last_valid_raw_price_map: Dict[str, float] = {}

        # 跨 Chunk 持久化活跃订单队列 [(oid, otype, side, sym_name, target_amount, limit_price, is_adj)]
        active_orders: List[Tuple] = []
        global_order_counter: List[int] = [0]
        last_day_key: Optional[Any] = None

        for chunk in chunk_stream:
            for s in chunk.symbols:
                if s not in all_symbols_seen:
                    all_symbols_seen.append(s)

            state = EngineState(
                n_steps=chunk.n_steps,
                n_symbols=chunk.n_symbols,
                initial_cash=current_cash,
                max_trades=self.max_trades,
            )

            # 精准按 Symbol 继承持仓状态
            for i, sym in enumerate(chunk.symbols):
                state.positions[i] = positions_map.get(sym, 0.0)
                state.avg_costs[i] = avg_costs_map.get(sym, 0.0)

            # T+1 可卖持仓初始化
            if self.enable_t1:
                available_positions = np.zeros(chunk.n_symbols, dtype=np.float64)
                for i, sym in enumerate(chunk.symbols):
                    available_positions[i] = available_positions_map.get(sym, 0.0)
            else:
                available_positions = state.positions

            cash_arr = np.array([current_cash], dtype=np.float64)

            vol_mat = chunk.volume if chunk.volume is not None else np.zeros((chunk.n_steps, chunk.n_symbols), dtype=np.float64)
            amt_mat = chunk.amount if chunk.amount is not None else np.zeros((chunk.n_steps, chunk.n_symbols), dtype=np.float64)
            is_tradable_mat = chunk.is_tradable if chunk.is_tradable is not None else np.ones((chunk.n_steps, chunk.n_symbols), dtype=np.bool_)

            if signals is not None and amounts is not None:
                # 极速向量模式
                full_sig = np.ascontiguousarray(signals, dtype=np.int8)
                full_amt = np.ascontiguousarray(amounts, dtype=np.float64)

                if full_sig.shape[0] > chunk.n_steps:
                    sig_mat = np.ascontiguousarray(full_sig[global_step_offset : global_step_offset + chunk.n_steps])
                    amt_matrix = np.ascontiguousarray(full_amt[global_step_offset : global_step_offset + chunk.n_steps])
                else:
                    sig_mat = full_sig
                    amt_matrix = full_amt

                adj_open_view = chunk.adj.open if hasattr(chunk, "adj") else chunk.open
                adj_close_view = chunk.adj.close if hasattr(chunk, "adj") else chunk.close

                run_engine_jit_kernel(
                    open_mat=chunk.open,
                    high_mat=chunk.high,
                    low_mat=chunk.low,
                    close_mat=chunk.close,
                    volume_mat=vol_mat,
                    amount_mat=amt_mat,
                    is_tradable_mat=is_tradable_mat,
                    signals_mat=sig_mat,
                    amounts_mat=amt_matrix,
                    matching_mode=self.matching_mode,
                    fee_rate=self.fee_rate,
                    min_fee=self.min_fee,
                    stamp_duty=self.stamp_duty,
                    slippage=self.slippage,
                    positions=state.positions,
                    avg_costs=state.avg_costs,
                    cash_arr=cash_arr,
                    portfolio_value=state.portfolio_value,
                    cash_history=state.cash_history,
                    trade_logs=state.trade_logs,
                    trade_count=state.trade_count,
                    max_volume_ratio=self.max_volume_ratio,
                    adj_open_mat=adj_open_view,
                    adj_close_mat=adj_close_view,
                    long_margin_ratio=self.long_margin_ratio,
                    short_margin_ratio=self.short_margin_ratio,
                    warmup_steps=warmup_steps,
                    lot_size=self.lot_size,
                    global_step_offset=global_step_offset,
                )

            elif strategy is not None:
                # Python 回调策略模式
                orders_buffer = []

                # 映射跨 Chunk 活跃订单至当前 Chunk 的 symbol_idx
                current_active_orders = []
                sym_to_idx = {s: i for i, s in enumerate(chunk.symbols)}
                for ord_item in active_orders:
                    oid, otype, side, sym_name, amt, lp, is_adj = ord_item
                    if sym_name in sym_to_idx:
                        s_idx = sym_to_idx[sym_name]
                        current_active_orders.append((oid, otype, side, s_idx, amt, lp, is_adj, sym_name))

                # 确定使用的复权视图
                close_view = chunk.adj.close if hasattr(chunk, "adj") else chunk.close
                open_view = chunk.adj.open if hasattr(chunk, "adj") else chunk.open
                high_view = chunk.adj.high if hasattr(chunk, "adj") else chunk.high
                low_view = chunk.adj.low if hasattr(chunk, "adj") else chunk.low

                ctx = BarContext(
                    step=0,
                    n_symbols=chunk.n_symbols,
                    timestamps=chunk.timestamps,
                    open_mat=chunk.open,
                    high_mat=chunk.high,
                    low_mat=chunk.low,
                    close_mat=chunk.close,
                    adj_close_mat=close_view,
                    adj_open_mat=open_view,
                    adj_high_mat=high_view,
                    adj_low_mat=low_view,
                    volume_mat=chunk.volume,
                    amount_mat=chunk.amount,
                    is_tradable_mat=chunk.is_tradable,
                    adj_factor=getattr(chunk, "adj_factor", None),
                    custom_fields=getattr(chunk, "custom_fields", None),
                    tables=getattr(chunk, "tables", None),
                    positions=state.positions,
                    available_positions=available_positions,
                    cash=cash_arr[0],
                    orders_buffer=orders_buffer,
                    warmup_steps=warmup_steps,
                    symbols=chunk.symbols,
                    order_counter_ref=global_order_counter,
                )

                for t in range(chunk.n_steps):
                    is_warmup = (t < warmup_steps)
                    global_step = global_step_offset + t

                    # 1. 自然日跨越判定 (Day-Rollover Detection) 与 T+1 解冻及计息
                    ts_val = chunk.timestamps[t]
                    if isinstance(ts_val, (int, np.integer)):
                        day_key = int(ts_val // 86_400_000)
                    else:
                        day_key = str(ts_val)[:10]

                    is_day_change = (last_day_key is not None and day_key != last_day_key)

                    if is_day_change:
                        # 跨日解锁 T+1 买入持仓为可卖持仓
                        if self.enable_t1:
                            available_positions[:] = state.positions[:]

                        # 扣除前一日融资融券利息 (按日结算，杜绝高频 Bar 重复扣除)
                        if not is_warmup:
                            daily_margin_r = self.margin_interest_rate / 252.0
                            daily_borrow_r = self.borrow_interest_rate / 252.0

                            if cash_arr[0] < 0.0 and daily_margin_r > 0.0:
                                cash_arr[0] -= abs(cash_arr[0]) * daily_margin_r

                            if daily_borrow_r > 0.0:
                                for i in range(chunk.n_symbols):
                                    if state.positions[i] < 0.0:
                                        curr_p = chunk.close[t, i]
                                        if np.isnan(curr_p) or curr_p <= 0:
                                            curr_p = last_valid_raw_price_map.get(chunk.symbols[i], state.avg_costs[i])
                                        if curr_p > 0:
                                            cash_arr[0] -= abs(state.positions[i]) * curr_p * daily_borrow_r

                    last_day_key = day_key

                    # 更新最新有效原始收盘价缓存
                    for i in range(chunk.n_symbols):
                        cp = chunk.close[t, i]
                        if not np.isnan(cp) and cp > 0:
                            last_valid_raw_price_map[chunk.symbols[i]] = float(cp)

                    # 2. 撤单处理: 从 current_active_orders 中移除已被标记撤销的订单 ID
                    if ctx.canceled_order_ids:
                        current_active_orders = [ord_item for ord_item in current_active_orders if ord_item[0] not in ctx.canceled_order_ids]

                    # 3. 撮合之前的未成交订单 (预热期不撮合)
                    if not is_warmup and len(current_active_orders) > 0:
                        remaining_active = []
                        for ord_item in current_active_orders:
                            oid, otype, side, sym_idx, target_amount, limit_price, is_adj, sym_name = ord_item

                            if otype == 1:
                                # 限价单撮合校验 (支持复权价换算为原始价格比较)
                                if is_adj:
                                    factor = (open_view[t, sym_idx] / chunk.open[t, sym_idx]) if (chunk.open[t, sym_idx] > 0 and open_view[t, sym_idx] > 0) else 1.0
                                    raw_limit_p = limit_price / factor
                                else:
                                    factor = (open_view[t, sym_idx] / chunk.open[t, sym_idx]) if (chunk.open[t, sym_idx] > 0 and open_view[t, sym_idx] > 0) else 1.0
                                    raw_limit_p = limit_price

                                is_trig, exec_p = check_limit_order_triggered_jit(
                                    side,
                                    raw_limit_p,
                                    chunk.open[t, sym_idx],
                                    chunk.high[t, sym_idx],
                                    chunk.low[t, sym_idx],
                                )
                                if is_trig:
                                    raw_p = exec_p
                                    adj_p = exec_p * factor
                                    vol_val = chunk.volume[t, sym_idx] if chunk.volume is not None else 0.0

                                    # T+1 卖出检查
                                    if side == -1 and self.enable_t1:
                                        if target_amount > available_positions[sym_idx]:
                                            target_amount = available_positions[sym_idx]

                                    if target_amount > 0:
                                        traded = execute_trade_jit(
                                            step_idx=global_step,
                                            symbol_idx=sym_idx,
                                            side=side,
                                            target_amount=target_amount,
                                            raw_price=raw_p,
                                            adj_price=adj_p,
                                            fee_rate=self.fee_rate,
                                            min_fee=self.min_fee,
                                            stamp_duty=self.stamp_duty,
                                            slippage=self.slippage,
                                            positions=state.positions,
                                            avg_costs=state.avg_costs,
                                            cash_arr=cash_arr,
                                            trade_logs=state.trade_logs,
                                            trade_count=state.trade_count,
                                            volume=vol_val,
                                            max_volume_ratio=self.max_volume_ratio,
                                            long_margin_ratio=self.long_margin_ratio,
                                            short_margin_ratio=self.short_margin_ratio,
                                            lot_size=self.lot_size,
                                        )
                                        if traded and side == -1 and self.enable_t1:
                                            available_positions[sym_idx] -= target_amount
                                else:
                                    remaining_active.append(ord_item)

                            elif otype == 0 and self.matching_mode == MATCHING_MODE_OPEN:
                                # OPEN 模式下上一步留存的市价单
                                raw_p = chunk.open[t, sym_idx]
                                adj_p = open_view[t, sym_idx]
                                vol_val = chunk.volume[t, sym_idx] if chunk.volume is not None else 0.0

                                if side == -1 and self.enable_t1:
                                    if target_amount > available_positions[sym_idx]:
                                        target_amount = available_positions[sym_idx]

                                if target_amount > 0:
                                    traded = execute_trade_jit(
                                        step_idx=global_step,
                                        symbol_idx=sym_idx,
                                        side=side,
                                        target_amount=target_amount,
                                        raw_price=raw_p,
                                        adj_price=adj_p,
                                        fee_rate=self.fee_rate,
                                        min_fee=self.min_fee,
                                        stamp_duty=self.stamp_duty,
                                        slippage=self.slippage,
                                        positions=state.positions,
                                        avg_costs=state.avg_costs,
                                        cash_arr=cash_arr,
                                        trade_logs=state.trade_logs,
                                        trade_count=state.trade_count,
                                        volume=vol_val,
                                        max_volume_ratio=self.max_volume_ratio,
                                        long_margin_ratio=self.long_margin_ratio,
                                        short_margin_ratio=self.short_margin_ratio,
                                        lot_size=self.lot_size,
                                    )
                                    if traded and side == -1 and self.enable_t1:
                                        available_positions[sym_idx] -= target_amount

                        current_active_orders = remaining_active

                    # 4. 运行当前 Bar 的策略逻辑 (预热期正常执行，供指标预热)
                    ctx.update_step(t, cash_arr[0], available_positions)
                    strategy(ctx)

                    # 5. 处理当前 Bar 新产生的订单 (预热期静默丢弃或清空)
                    if not is_warmup and len(orders_buffer) > 0:
                        for raw_order in orders_buffer:
                            # 规范化订单格式
                            if len(raw_order) == 3:
                                side, sym_idx, target_amount = raw_order
                                oid, otype, limit_price, is_adj = 0, 0, 0.0, True
                            elif len(raw_order) == 6:
                                oid, otype, side, sym_idx, target_amount, limit_price = raw_order
                                is_adj = True
                            else:
                                oid, otype, side, sym_idx, target_amount, limit_price, is_adj = raw_order

                            sym_name = chunk.symbols[sym_idx]

                            if otype == 1:
                                # 新产生的限价单
                                if is_adj:
                                    factor = (open_view[t, sym_idx] / chunk.open[t, sym_idx]) if (chunk.open[t, sym_idx] > 0 and open_view[t, sym_idx] > 0) else 1.0
                                    raw_limit_p = limit_price / factor
                                else:
                                    factor = (open_view[t, sym_idx] / chunk.open[t, sym_idx]) if (chunk.open[t, sym_idx] > 0 and open_view[t, sym_idx] > 0) else 1.0
                                    raw_limit_p = limit_price

                                is_trig, exec_p = check_limit_order_triggered_jit(
                                    side,
                                    raw_limit_p,
                                    chunk.open[t, sym_idx],
                                    chunk.high[t, sym_idx],
                                    chunk.low[t, sym_idx],
                                )
                                if is_trig:
                                    raw_p = exec_p
                                    adj_p = exec_p * factor
                                    vol_val = chunk.volume[t, sym_idx] if chunk.volume is not None else 0.0

                                    if side == -1 and self.enable_t1:
                                        if target_amount > available_positions[sym_idx]:
                                            target_amount = available_positions[sym_idx]

                                    if target_amount > 0:
                                        traded = execute_trade_jit(
                                            step_idx=global_step,
                                            symbol_idx=sym_idx,
                                            side=side,
                                            target_amount=target_amount,
                                            raw_price=raw_p,
                                            adj_price=adj_p,
                                            fee_rate=self.fee_rate,
                                            min_fee=self.min_fee,
                                            stamp_duty=self.stamp_duty,
                                            slippage=self.slippage,
                                            positions=state.positions,
                                            avg_costs=state.avg_costs,
                                            cash_arr=cash_arr,
                                            trade_logs=state.trade_logs,
                                            trade_count=state.trade_count,
                                            volume=vol_val,
                                            max_volume_ratio=self.max_volume_ratio,
                                            long_margin_ratio=self.long_margin_ratio,
                                            short_margin_ratio=self.short_margin_ratio,
                                            lot_size=self.lot_size,
                                        )
                                        if traded and side == -1 and self.enable_t1:
                                            available_positions[sym_idx] -= target_amount
                                else:
                                    current_active_orders.append((oid, otype, side, sym_idx, target_amount, limit_price, is_adj, sym_name))

                            else:
                                # 市价单
                                if self.matching_mode == MATCHING_MODE_OPEN:
                                    current_active_orders.append((oid, otype, side, sym_idx, target_amount, limit_price, is_adj, sym_name))
                                else:
                                    raw_p = get_execution_price(
                                        self.matching_mode,
                                        chunk.open[t, sym_idx],
                                        chunk.high[t, sym_idx],
                                        chunk.low[t, sym_idx],
                                        chunk.close[t, sym_idx],
                                        vol_mat[t, sym_idx],
                                        amt_mat[t, sym_idx],
                                    )
                                    adj_p = get_execution_price(
                                        self.matching_mode,
                                        open_view[t, sym_idx],
                                        high_view[t, sym_idx],
                                        low_view[t, sym_idx],
                                        close_view[t, sym_idx],
                                        vol_mat[t, sym_idx],
                                        amt_mat[t, sym_idx],
                                    )
                                    vol_val = chunk.volume[t, sym_idx] if chunk.volume is not None else 0.0

                                    if side == -1 and self.enable_t1:
                                        if target_amount > available_positions[sym_idx]:
                                            target_amount = available_positions[sym_idx]

                                    if target_amount > 0:
                                        traded = execute_trade_jit(
                                            step_idx=global_step,
                                            symbol_idx=sym_idx,
                                            side=side,
                                            target_amount=target_amount,
                                            raw_price=raw_p,
                                            adj_price=adj_p,
                                            fee_rate=self.fee_rate,
                                            min_fee=self.min_fee,
                                            stamp_duty=self.stamp_duty,
                                            slippage=self.slippage,
                                            positions=state.positions,
                                            avg_costs=state.avg_costs,
                                            cash_arr=cash_arr,
                                            trade_logs=state.trade_logs,
                                            trade_count=state.trade_count,
                                            volume=vol_val,
                                            max_volume_ratio=self.max_volume_ratio,
                                            long_margin_ratio=self.long_margin_ratio,
                                            short_margin_ratio=self.short_margin_ratio,
                                            lot_size=self.lot_size,
                                        )
                                        if traded and side == -1 and self.enable_t1:
                                            available_positions[sym_idx] -= target_amount

                        orders_buffer.clear()
                    else:
                        orders_buffer.clear()

                    state.cash = cash_arr[0]
                    state.cash_history[t] = state.cash

                    # 计算准确的逐步净资产 (停牌股回退至最后有效原始收盘价)
                    pos_val = 0.0
                    for i in range(chunk.n_symbols):
                        if state.positions[i] != 0.0:
                            curr_price = chunk.close[t, i]
                            if np.isnan(curr_price) or curr_price <= 0:
                                curr_price = last_valid_raw_price_map.get(chunk.symbols[i], state.avg_costs[i])
                            pos_val += state.positions[i] * curr_price

                    state.portfolio_value[t] = state.cash + pos_val

                # 更新跨 Chunk 活跃订单队列
                active_orders = [(o[0], o[1], o[2], o[7], o[4], o[5], o[6]) for o in current_active_orders]

            # 收集该 Chunk 的运行记录
            all_portfolio_values.append(state.portfolio_value)
            all_cash_histories.append(state.cash_history)
            all_timestamps.append(chunk.timestamps)

            t_count = int(state.trade_count[0])
            if t_count > 0:
                chunk_trades = state.trade_logs[:t_count].copy()
                # 将该 Chunk 的局部 symbol_idx 映射至权威全局 all_symbols_seen 索引
                chunk_sym_to_global = {local_i: all_symbols_seen.index(s) for local_i, s in enumerate(chunk.symbols)}
                for trade_i in range(t_count):
                    local_s_idx = int(chunk_trades[trade_i, 1])
                    if local_s_idx in chunk_sym_to_global:
                        chunk_trades[trade_i, 1] = float(chunk_sym_to_global[local_s_idx])
                all_trade_logs.append(chunk_trades)

            # 提取期末状态写入全局字典以供下一个 Chunk 继承
            current_cash = float(cash_arr[0])
            for i, sym in enumerate(chunk.symbols):
                if state.positions[i] != 0.0:
                    positions_map[sym] = float(state.positions[i])
                    avg_costs_map[sym] = float(state.avg_costs[i])
                    if self.enable_t1:
                        available_positions_map[sym] = float(available_positions[i])
                else:
                    positions_map.pop(sym, None)
                    avg_costs_map.pop(sym, None)
                    available_positions_map.pop(sym, None)

            global_step_offset += chunk.n_steps

        # 合并多 Chunk 结果
        full_pv = np.concatenate(all_portfolio_values) if all_portfolio_values else np.array([])
        full_cash = np.concatenate(all_cash_histories) if all_cash_histories else np.array([])
        full_timestamps = np.concatenate(all_timestamps) if all_timestamps else np.array([])

        if all_trade_logs:
            full_trade_logs = np.vstack(all_trade_logs)
            full_trade_count = len(full_trade_logs)
        else:
            full_trade_logs = np.zeros((0, 7), dtype=np.float64)
            full_trade_count = 0

        final_symbols = all_symbols_seen if all_symbols_seen else []

        return BacktestResult(
            trade_logs_mat=full_trade_logs,
            trade_count=full_trade_count,
            portfolio_value=full_pv,
            cash_history=full_cash,
            timestamps=full_timestamps,
            symbols=final_symbols,
            initial_cash=self.initial_cash,
            annualization_factor=self.annualization_factor,
        )

    def run_fast(
        self,
        signals: np.ndarray,
        amounts: np.ndarray,
        data: Union[dict, MarketData, Iterable[MarketData], Any],
        warmup_steps: int = 0,
    ) -> BacktestResult:
        """
        向后兼容的向量化快捷运行方法
        """
        return self.run(signals=signals, amounts=amounts, data=data, warmup_steps=warmup_steps)
