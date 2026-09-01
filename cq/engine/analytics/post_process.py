"""
结果 Commit 与后处理分析模块 (PostProcess)

将预分配大数组交易日志切片导出为 Polars DataFrame，并计算夏普比率、最大回撤等回测绩效指标。
"""

from typing import Dict, List, Any
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
    回测结果对象
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
        annualization_factor: int = 252,
    ):
        self.trade_count = trade_count
        self.portfolio_value = portfolio_value
        self.cash_history = cash_history
        self.timestamps = timestamps
        self.symbols = symbols
        self.initial_cash = initial_cash
        self.annualization_factor = annualization_factor

        # 仅截取有效成交记录 [0 : trade_count]
        valid_trades = trade_logs_mat[:trade_count]

        if trade_count > 0:
            step_indices = valid_trades[:, 0].astype(int)
            stock_indices = valid_trades[:, 1].astype(int)

            trade_datetimes = [_format_ts_str(timestamps[i]) for i in step_indices]
            trade_symbols = [symbols[i] for i in stock_indices]
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

    def calc_metrics(self) -> Dict[str, Any]:
        """计算核心绩效指标"""
        pv = self.portfolio_value
        if len(pv) == 0:
            return {}

        total_return = (pv[-1] - self.initial_cash) / self.initial_cash
        n_periods = len(pv)

        # 逐日/逐 Bar 收益率
        returns = np.diff(pv) / pv[:-1]
        returns = np.nan_to_num(returns, nan=0.0)

        # 年化收益率
        if n_periods > 1:
            annualized_return = (1.0 + total_return) ** (self.annualization_factor / n_periods) - 1.0
        else:
            annualized_return = 0.0

        # 波动率与夏普比率
        std_ret = np.std(returns)
        if std_ret > 1e-8:
            sharpe_ratio = (np.mean(returns) / std_ret) * np.sqrt(self.annualization_factor)
        else:
            sharpe_ratio = 0.0

        # 最大回撤 (Max Drawdown)
        cummax = np.maximum.accumulate(pv)
        drawdowns = (cummax - pv) / cummax
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # 胜率与交易统计
        total_trades = self.trade_count
        total_fee = float(self.trade_logs["fee"].sum()) if total_trades > 0 else 0.0

        return {
            "initial_cash": self.initial_cash,
            "final_value": float(pv[-1]),
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "total_trades": total_trades,
            "total_fee": float(total_fee),
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
            f"  最大回撤 (Max Drawdown):     {m.get('max_drawdown', 0.0) * 100:.2f}%",
            f"  总交易笔数 (Total Trades):   {m.get('total_trades', 0)}",
            f"  总手续费/印花税 (Total Fee): {m.get('total_fee', 0.0):,.2f}",
            "=======================================================================",
        ]
        return "\n".join(lines)
