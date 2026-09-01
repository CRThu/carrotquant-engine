"""
市价/限价撮合引擎内核算子 (@njit)

提供纯 JIT 编译的带费率、印花税、滑点与多种撮合价格模式的通用订单撮合逻辑。
"""

from enum import IntEnum
import numpy as np
from numba import njit

# 撮合价格模式常量与 Enum
MATCHING_MODE_OPEN = 0
MATCHING_MODE_CLOSE = 1
MATCHING_MODE_VWAP = 2
MATCHING_MODE_TWAP = 3


class MatchingMode(IntEnum):
    """撮合模式枚举"""
    OPEN = MATCHING_MODE_OPEN
    CLOSE = MATCHING_MODE_CLOSE
    VWAP = MATCHING_MODE_VWAP
    TWAP = MATCHING_MODE_TWAP

    @classmethod
    def parse(cls, val):
        """解析字符串或 Enum/Int 撮合模式"""
        if isinstance(val, cls):
            return val.value
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            val_upper = val.strip().upper()
            if val_upper == "OPEN":
                return MATCHING_MODE_OPEN
            elif val_upper == "CLOSE":
                return MATCHING_MODE_CLOSE
            elif val_upper == "VWAP":
                return MATCHING_MODE_VWAP
            elif val_upper == "TWAP":
                return MATCHING_MODE_TWAP
        raise ValueError(f"无法解析的撮合模式: {val}")


@njit(fastmath=True, nogil=True)
def check_limit_order_triggered_jit(
    side: int,
    limit_price: float,
    open_p: float,
    high_p: float,
    low_p: float,
) -> tuple:
    """
    检查限价单在当前 Bar 是否触发成交。
    
    买入限价单 (side == 1): 当市场最低价 low_p <= limit_price 时触发，成交价格为 min(limit_price, open_p)
    卖出限价单 (side == -1): 当市场最高价 high_p >= limit_price 时触发，成交价格为 max(limit_price, open_p)
    """
    if np.isnan(limit_price) or limit_price <= 0.0 or np.isnan(open_p) or open_p <= 0.0:
        return False, 0.0

    if side == 1:
        if limit_price >= low_p:
            exec_p = min(limit_price, open_p) if open_p > 0 else limit_price
            return True, exec_p
    elif side == -1:
        if limit_price <= high_p:
            exec_p = max(limit_price, open_p) if open_p > 0 else limit_price
            return True, exec_p

    return False, 0.0


@njit(fastmath=True, nogil=True)
def get_execution_price(
    mode: int,
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    volume: float,
    amount: float,
) -> float:
    """根据指定的撮合价格模式计算执行价格"""
    if mode == MATCHING_MODE_OPEN:
        return open_p
    elif mode == MATCHING_MODE_CLOSE:
        return close_p
    elif mode == MATCHING_MODE_VWAP:
        if volume > 0.0:
            return amount / volume
        return close_p
    elif mode == MATCHING_MODE_TWAP:
        return (high_p + low_p + close_p) / 3.0
    else:
        return close_p


@njit(fastmath=True, nogil=True)
def execute_trade_jit(
    step_idx: int,
    symbol_idx: int,
    side: int,  # 1 为买入(增加持仓), -1 为卖出(减少持仓/做空)
    target_amount: float,
    raw_price: float,
    adj_price: float,
    fee_rate: float,
    min_fee: float,
    stamp_duty: float,
    slippage: float,
    positions: np.ndarray,
    avg_costs: np.ndarray,
    cash_arr: np.ndarray,  # 长度为 1 的 1D 数组方便原地更新 cash
    trade_logs: np.ndarray,
    trade_count: np.ndarray,
    volume: float = 0.0,
    max_volume_ratio: float = 1.0,
    long_margin_ratio: float = 1.0,
    short_margin_ratio: float = 1.0,
    lot_size: int = 1,
) -> bool:
    """
    JIT 极速撮合单笔交易 (支持多空双向、盘口成交量限制、做多/做空保证金率校验及整手约束)。

    Returns:
        bool: 是否成功成交
    """
    if target_amount <= 0.0 or np.isnan(raw_price) or raw_price <= 0.0:
        return False

    # 买入整手约束 (Lot Size Constraint)
    if side == 1 and lot_size > 1:
        target_amount = np.floor(target_amount / lot_size) * lot_size
        if target_amount <= 0.0:
            return False

    # 盘口成交量上限约束 (Max Volume Ratio Constraint)
    if volume > 0.0 and max_volume_ratio < 1.0:
        max_tradable = volume * max_volume_ratio
        if side == 1 and lot_size > 1:
            max_tradable = np.floor(max_tradable / lot_size) * lot_size
        if target_amount > max_tradable:
            target_amount = max_tradable
            if target_amount <= 0.0:
                return False

    current_cash = cash_arr[0]
    curr_pos = positions[symbol_idx]

    # 计算考虑滑点后的真实执行价格
    if side == 1:  # 买入方向
        exec_raw_price = raw_price * (1.0 + slippage)
        exec_adj_price = adj_price * (1.0 + slippage)
    else:  # 卖出方向
        exec_raw_price = raw_price * (1.0 - slippage)
        exec_adj_price = adj_price * (1.0 - slippage)

    if side == 1:  # 买入逻辑 (pos += target_amount)
        raw_trade_value = target_amount * exec_raw_price
        comm = max(raw_trade_value * fee_rate, min_fee) if fee_rate > 0 else 0.0
        # 考虑到做多保证金率 long_margin_ratio
        req_margin = raw_trade_value * long_margin_ratio + comm

        # 校验现金/保证金是否充足，若不足则反算可买数量
        if req_margin > current_cash:
            if current_cash <= min_fee:
                return False
            # 重新反算按保证金率可买数量
            denom = exec_raw_price * (long_margin_ratio + fee_rate)
            if denom <= 0.0:
                return False
            target_amount = (current_cash - min_fee) / denom
            if lot_size > 1:
                target_amount = np.floor(target_amount / lot_size) * lot_size
            if target_amount <= 0.0:
                return False
            raw_trade_value = target_amount * exec_raw_price
            comm = max(raw_trade_value * fee_rate, min_fee) if fee_rate > 0 else 0.0

        total_cash_outflow = raw_trade_value + comm

        # 更新持仓与开仓均价
        new_pos = curr_pos + target_amount
        if curr_pos >= 0.0:
            if new_pos > 0.0:
                avg_costs[symbol_idx] = (curr_pos * avg_costs[symbol_idx] + target_amount * exec_adj_price) / new_pos
        else:
            # 此前为空头，买入属于平空/反手
            if new_pos > 0.0:
                avg_costs[symbol_idx] = exec_adj_price
            elif abs(new_pos) < 1e-8:
                avg_costs[symbol_idx] = 0.0

        positions[symbol_idx] = new_pos
        cash_arr[0] -= total_cash_outflow
        paid_fee = comm

    else:  # 卖出逻辑 (side == -1, pos -= target_amount，天然支持做空)
        raw_trade_value = target_amount * exec_raw_price
        comm = max(raw_trade_value * fee_rate, min_fee) if fee_rate > 0 else 0.0
        duty = raw_trade_value * stamp_duty
        total_fee = comm + duty
        net_proceeds = raw_trade_value - total_fee

        # 卖空/反手空头保证金校验
        if curr_pos <= 0.0:
            # 纯加空/新开空
            req_short_margin = raw_trade_value * short_margin_ratio + total_fee
            if req_short_margin > current_cash and short_margin_ratio > 0:
                if current_cash <= total_fee:
                    return False
                target_amount = (current_cash - total_fee) / (exec_raw_price * short_margin_ratio)
                if target_amount <= 0.0:
                    return False
                raw_trade_value = target_amount * exec_raw_price
                comm = max(raw_trade_value * fee_rate, min_fee) if fee_rate > 0 else 0.0
                duty = raw_trade_value * stamp_duty
                total_fee = comm + duty
                net_proceeds = raw_trade_value - total_fee
        elif target_amount > curr_pos:
            # 此前为多头，卖出量大于多头持仓 -> 多翻空反手
            short_part = target_amount - curr_pos
            close_part = curr_pos
            close_val = close_part * exec_raw_price
            close_comm = max(close_val * fee_rate, min_fee) if fee_rate > 0 else 0.0
            close_duty = close_val * stamp_duty
            close_net = close_val - (close_comm + close_duty)
            cash_after_close = current_cash + close_net

            short_val = short_part * exec_raw_price
            short_comm = max(short_val * fee_rate, 0.0)
            short_duty = short_val * stamp_duty
            req_short_margin = short_val * short_margin_ratio + (short_comm + short_duty)

            if req_short_margin > cash_after_close and short_margin_ratio > 0:
                max_short = (cash_after_close - min_fee) / (exec_raw_price * short_margin_ratio)
                if max_short < 0.0:
                    max_short = 0.0
                target_amount = curr_pos + max_short
                if target_amount <= 0.0:
                    return False
                raw_trade_value = target_amount * exec_raw_price
                comm = max(raw_trade_value * fee_rate, min_fee) if fee_rate > 0 else 0.0
                duty = raw_trade_value * stamp_duty
                total_fee = comm + duty
                net_proceeds = raw_trade_value - total_fee

        new_pos = curr_pos - target_amount
        if curr_pos <= 0.0:
            if new_pos < 0.0:
                # 加空
                old_abs_pos = abs(curr_pos)
                avg_costs[symbol_idx] = (old_abs_pos * avg_costs[symbol_idx] + target_amount * exec_adj_price) / abs(new_pos)
        else:
            # 此前为多头，卖出属于平多/反手
            if new_pos < 0.0:
                avg_costs[symbol_idx] = exec_adj_price
            elif abs(new_pos) < 1e-8:
                avg_costs[symbol_idx] = 0.0

        positions[symbol_idx] = new_pos
        if abs(positions[symbol_idx]) < 1e-8:
            positions[symbol_idx] = 0.0
            avg_costs[symbol_idx] = 0.0

        cash_arr[0] += net_proceeds
        paid_fee = total_fee

    # 写入预分配交易日志
    idx = trade_count[0]
    if idx < trade_logs.shape[0]:
        trade_logs[idx, 0] = float(step_idx)
        trade_logs[idx, 1] = float(symbol_idx)
        trade_logs[idx, 2] = float(side)
        trade_logs[idx, 3] = target_amount
        trade_logs[idx, 4] = exec_adj_price
        trade_logs[idx, 5] = paid_fee
        trade_logs[idx, 6] = cash_arr[0]
        trade_count[0] += 1

    return True


