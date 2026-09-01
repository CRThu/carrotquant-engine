"""
结果 Commit 与后处理分析模块 (PostProcess)

将预分配大数组交易日志切片导出为 Polars DataFrame，并计算夏普比率、最大回撤等回测绩效指标。
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import polars as pl


def _format_ts_str(val: Any) -> str:
    """将时间戳 (int64 ms 或 string) 转换为可读格式字符串"""
    if isinstance(val, (int, np.integer)):
        try:
            dt = datetime.fromtimestamp(int(val) / 1000.0, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(val)
    return str(val)


class BacktestResult:
    """
    回测结果对象 (BacktestResult)
    """

    def __init__(
        self,
        trade_logs_mat: np.ndarray,
        trade_count: int,
        portfolio_value: np.ndarray,
        cash_history: np.ndarray,
        timestamps: np.ndarray,
        symbols: List[str],
        initial_cash: float,
        annualization_factor: Optional[int] = None,
    ):
        self.trade_count = trade_count
        self.portfolio_value = np.ascontiguousarray(portfolio_value, dtype=np.float64)
        self.cash_history = np.ascontiguousarray(cash_history, dtype=np.float64)
        self.timestamps = np.ascontiguousarray(timestamps)
        self.symbols = list(symbols)
        self.initial_cash = float(initial_cash)

        # 智能自适应年化因子 (若未显式指定)
        if annualization_factor is not None:
            self.annualization_factor = int(annualization_factor)
        else:
            self.annualization_factor = self._infer_annualization_factor(timestamps)

        # 仅截取有效成交记录 [0 : trade_count]
        valid_trades = trade_logs_mat[:trade_count]

        if trade_count > 0:
            step_indices = valid_trades[:, 0].astype(int)
            stock_indices = valid_trades[:, 1].astype(int)

            trade_datetimes = [_format_ts_str(timestamps[i]) if i < len(timestamps) else str(i) for i in step_indices]
            trade_symbols = [symbols[i] if i < len(symbols) else f"SYM_{i}" for i in stock_indices]
            side_strs = ["BUY" if s > 0 else "SELL" for s in valid_trades[:, 2]]

            self.trade_logs = pl.DataFrame({
                "step_idx": step_indices,
                "datetime": trade_datetimes,
                "symbol": trade_symbols,
                "side": side_strs,
                "amount": valid_trades[:, 3],
                "price": valid_trades[:, 4],
                "fee": valid_trades[:, 5],
                "cash_after": valid_trades[:, 6],
            })
        else:
            self.trade_logs = pl.DataFrame({
                "step_idx": pl.Series(dtype=pl.Int64),
                "datetime": pl.Series(dtype=pl.String),
                "symbol": pl.Series(dtype=pl.String),
                "side": pl.Series(dtype=pl.String),
                "amount": pl.Series(dtype=pl.Float64),
                "price": pl.Series(dtype=pl.Float64),
                "fee": pl.Series(dtype=pl.Float64),
                "cash_after": pl.Series(dtype=pl.Float64),
            })

        # 构建资产曲线 DataFrame
        self.portfolio_df = pl.DataFrame({
            "datetime": [_format_ts_str(t) for t in timestamps],
            "cash": cash_history,
            "portfolio_value": portfolio_value,
        })

    @staticmethod
    def _infer_annualization_factor(timestamps: np.ndarray) -> int:
        """根据时间戳中位数间隔智能推断年化周期因子"""
        if len(timestamps) < 3:
            return 252

        sample_ts = timestamps[: min(len(timestamps), 500)]
        if np.issubdtype(sample_ts.dtype, np.number):
            diffs = np.diff(sample_ts.astype(np.float64))
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                med_ms = np.median(diffs)
                if med_ms <= 65_000:       # 1m 分钟线
                    return 252 * 240
                elif med_ms <= 310_000:    # 5m 分钟线
                    return 252 * 48
                elif med_ms <= 950_000:    # 15m
                    return 252 * 16
                elif med_ms <= 1_900_000:  # 30m
                    return 252 * 8
                elif med_ms <= 3_700_000:  # 60m / 1h
                    return 252 * 4
        return 252  # 默认日线 252 交易日

    def _calc_trade_performance(self) -> Dict[str, Any]:
        """计算逐笔 FIFO 匹配的平仓盈亏、胜率、盈亏比与最大连续亏损次数"""
        if self.trade_count == 0 or self.trade_logs.is_empty():
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "realized_pnl": 0.0,
                "win_trades": 0,
                "loss_trades": 0,
                "max_consecutive_losses": 0,
            }

        symbol_positions: Dict[str, List[Tuple[float, float]]] = {}  # {symbol: [(amount, price)]}
        realized_pnls: List[float] = []

        rows = self.trade_logs.to_dicts()
        for r in rows:
            sym = r["symbol"]
            side = r["side"]
            amt = float(r["amount"])
            price = float(r["price"])
            fee = float(r["fee"])

            if sym not in symbol_positions:
                symbol_positions[sym] = []

            pos_queue = symbol_positions[sym]

            if side == "BUY":
                # 买入开多 / 平空
                rem_amt = amt
                while rem_amt > 1e-8 and pos_queue and pos_queue[0][0] < 0:
                    short_amt, short_p = pos_queue[0]
                    match_amt = min(rem_amt, abs(short_amt))
                    pnl = match_amt * (short_p - price) - (fee * (match_amt / amt))
                    realized_pnls.append(pnl)
                    rem_amt -= match_amt
                    if abs(short_amt) - match_amt < 1e-8:
                        pos_queue.pop(0)
                    else:
                        pos_queue[0] = (short_amt + match_amt, short_p)

                if rem_amt > 1e-8:
                    pos_queue.append((rem_amt, price))

            else:  # SELL
                # 卖出平多 / 开空
                rem_amt = amt
                while rem_amt > 1e-8 and pos_queue and pos_queue[0][0] > 0:
                    long_amt, long_p = pos_queue[0]
                    match_amt = min(rem_amt, long_amt)
                    pnl = match_amt * (price - long_p) - (fee * (match_amt / amt))
                    realized_pnls.append(pnl)
                    rem_amt -= match_amt
                    if long_amt - match_amt < 1e-8:
                        pos_queue.pop(0)
                    else:
                        pos_queue[0] = (long_amt - match_amt, long_p)

                if rem_amt > 1e-8:
                    pos_queue.append((-rem_amt, price))

        if not realized_pnls:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "realized_pnl": 0.0,
                "win_trades": 0,
                "loss_trades": 0,
                "max_consecutive_losses": 0,
            }

        wins = [p for p in realized_pnls if p > 0]
        losses = [p for p in realized_pnls if p < 0]
        win_rate = len(wins) / len(realized_pnls) if realized_pnls else 0.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-8 else (float("inf") if gross_profit > 0 else 0.0)

        # 最大连续亏损次数
        cur_loss_streak = 0
        max_loss_streak = 0
        for p in realized_pnls:
            if p < 0:
                cur_loss_streak += 1
                if cur_loss_streak > max_loss_streak:
                    max_loss_streak = cur_loss_streak
            else:
                cur_loss_streak = 0

        return {
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "realized_pnl": float(sum(realized_pnls)),
            "win_trades": len(wins),
            "loss_trades": len(losses),
            "max_consecutive_losses": max_loss_streak,
        }

    def calc_metrics(self) -> Dict[str, Any]:
        """计算全面专业回测绩效指标"""
        pv = self.portfolio_value
        if len(pv) == 0:
            return {}

        total_return = (pv[-1] - self.initial_cash) / self.initial_cash
        n_periods = len(pv)

        # 逐日/逐 Bar 收益率
        safe_pv_denom = np.where(pv[:-1] != 0, pv[:-1], np.nan)
        returns = np.diff(pv) / safe_pv_denom
        returns = np.nan_to_num(returns, nan=0.0)

        # 年化收益率 (包含穿仓/负资产保护)
        if total_return <= -1.0:
            annualized_return = -1.0
        elif n_periods > 1:
            annualized_return = (1.0 + total_return) ** (self.annualization_factor / n_periods) - 1.0
        else:
            annualized_return = 0.0

        # 波动率与夏普比率
        std_ret = np.std(returns)
        annualized_volatility = float(std_ret * np.sqrt(self.annualization_factor))
        if std_ret > 1e-8:
            sharpe_ratio = float((np.mean(returns) / std_ret) * np.sqrt(self.annualization_factor))
        else:
            sharpe_ratio = 0.0

        # 索提诺比率 (Sortino Ratio)
        downside_returns = returns[returns < 0]
        downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0
        if downside_std > 1e-8:
            sortino_ratio = float((np.mean(returns) / downside_std) * np.sqrt(self.annualization_factor))
        else:
            sortino_ratio = 0.0

        # 最大回撤 (Max Drawdown)
        cummax = np.maximum.accumulate(np.maximum(pv, 0.0))
        drawdowns = np.where(cummax > 0, (cummax - pv) / cummax, 1.0)
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # 卡玛比率 (Calmar Ratio)
        calmar_ratio = float(annualized_return / max_drawdown) if max_drawdown > 1e-8 else 0.0

        # 逐笔交易统计
        trade_perf = self._calc_trade_performance()
        total_trades = self.trade_count
        total_fee = float(self.trade_logs["fee"].sum()) if total_trades > 0 else 0.0

        return {
            "initial_cash": self.initial_cash,
            "final_value": float(pv[-1]),
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "total_trades": total_trades,
            "win_rate": trade_perf["win_rate"],
            "profit_factor": trade_perf["profit_factor"],
            "max_consecutive_losses": trade_perf["max_consecutive_losses"],
            "total_fee": total_fee,
            "annualization_factor": self.annualization_factor,
        }

    def summary(self) -> str:
        """生成格式化回测绩效文本报告"""
        m = self.calc_metrics()
        lines = [
            "=================== 回测绩效报告 (Backtest Summary) ===================",
            f"  初始资金 (Initial Cash):     {m.get('initial_cash', 0.0):,.2f}",
            f"  最终资产 (Final Value):      {m.get('final_value', 0.0):,.2f}",
            f"  累计收益率 (Total Return):    {m.get('total_return', 0.0) * 100:.2f}%",
            f"  年化收益率 (Annual Return):   {m.get('annualized_return', 0.0) * 100:.2f}%",
            f"  夏普比率 (Sharpe Ratio):     {m.get('sharpe_ratio', 0.0):.4f}",
            f"  索提诺比率 (Sortino Ratio):   {m.get('sortino_ratio', 0.0):.4f}",
            f"  最大回撤 (Max Drawdown):     {m.get('max_drawdown', 0.0) * 100:.2f}%",
            f"  卡玛比率 (Calmar Ratio):     {m.get('calmar_ratio', 0.0):.4f}",
            f"  胜率 (Win Rate):             {m.get('win_rate', 0.0) * 100:.2f}%",
            f"  盈亏比 (Profit Factor):      {m.get('profit_factor', 0.0):.2f}",
            f"  最大连续亏损 (Consec Loss):  {m.get('max_consecutive_losses', 0)}",
            f"  总交易笔数 (Total Trades):   {m.get('total_trades', 0)}",
            f"  总手续费/印花税 (Total Fee): {m.get('total_fee', 0.0):,.2f}",
            "=======================================================================",
        ]
        return "\n".join(lines)
