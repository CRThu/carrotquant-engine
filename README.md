# CarrotQuant Engine (`carrotquant-engine`)

[![PyPI version](https://img.shields.io/pypi/v/carrotquant-engine.svg)](https://pypi.org/project/carrotquant-engine/)
[![Python Version](https://img.shields.io/badge/python-%3E%3D3.12-blue)](https://pypi.org/project/carrotquant-engine/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

**CarrotQuant Engine** 是基于 Python 与 Numba 的高性能事件驱动与向量化量化回测引擎，支持全市场多品种多表的数据供给、撮合执行与绩效分析。

## 📦 安装指南 (Installation)

环境要求：**Python >= 3.12**（支持 Python 3.12 / 3.13 / 3.14+）。

```bash
# 使用 uv 安装
uv add carrotquant-engine
```

## 🛠️ 特性 (Features)
- **高性能计算内核**：基于 Numba JIT 与连续 2D C-Contiguous 内存布局，降低循环执行与内存分配开销。
- **Duck Typing 数据协议与多表供给**：`engine.run(data=...)` 原生支持标准 `dict` 传参与 Duck Typing（具备 `.to_df()` / `.read()` / `.collect()` 的数据源、`LazyFrame`、`DataFrame` 或 Stream 生成器），支持主行情表、副行情表（如指数 `index`）、特征列（如 `pe_ttm`）、稀疏离散事件（如龙虎榜）与静态属性（如板块）。
- **单趟极速矩阵构建 (MatrixBuilder)**：单趟完成全局坐标映射与内存填充，严格校验 OHLC 全量价格列，未提供 `volume`/`amount` 时保持纯净 `None`。
- **Master Clock 时空对齐**：副 TS 表按主时钟自动 Left Join 内存对齐，超出时间步截断，缺失时间步填充 `NaN`。
- **多空双向撮合**：`buy` / `sell` 支持做多与做空 (`pos += amount` 与 `pos -= amount`)，统一浮动资产计算 $PV = \text{Cash} + \sum \text{pos}_i \times \text{close}_i$。
- **轻量动态复权**：`data.close` 为原始成交价（用于资金交割），`data.adj.close` / `ctx.adj.close` 提供动态后复权视图。
- **防未来函数切片**：策略通过 `ctx.get('factor')`（当前 $t$ 步快照）与 `ctx.get_history('factor')`（物理边界 `[:t+1, :]`）访问数据，避免未来数据泄露。
- **分块流式预热 (`warmup_steps`)**：支持分块流式回测并在预热期只更新指标状态而不触发资金扣除。
- **流动性与撮合限制**：支持 `max_volume_ratio`（盘口成交量比例限制）、限价单 `buy_limit` / `sell_limit` 与 `cancel_order` 撤单机制。
- **保证金与融资融券费率**：支持设置 `long_margin_ratio` / `short_margin_ratio`（保证金率校验），以及 `margin_interest_rate` / `borrow_interest_rate`（日频利息计提）。

## 🚀 快速开始

```python
from cq.engine import strategy, BarContext, Engine, ts_table, event_table, static_table
import polars as pl

# 1. 定义策略 (支持访问副行情表 index 与离散事件表)
@strategy
def multi_asset_strategy(ctx: BarContext):
    # 读取副表（指数）当前价格与历史收盘价切片 [:t+1, :]
    index_table = ctx.get("index")
    index_close_hist = index_table.close_history
    
    # 读取离散事件表 (龙虎榜)
    dt_events = ctx.get("dragon_tiger")
    
    for i in range(ctx.n_symbols):
        if not ctx.is_tradable[i]:
            continue

        # 使用 ctx.adj.close_history 读取后复权历史收盘价
        c_hist = ctx.adj.close_history[-20:, i]
        ma5 = c_hist[-5:].mean()
        ma20 = c_hist[-20:].mean()

        # 结合指数趋势与均线信号买卖
        if ma5 > ma20 and ctx.positions[i] == 0:
            ctx.buy(symbol_idx=i, amount=100)
        elif ma5 < ma20 and ctx.positions[i] > 0:
            ctx.sell(symbol_idx=i, amount=ctx.positions[i])

# 2. 初始化回测引擎
engine = Engine(
    initial_cash=1_000_000.0,
    fee_rate=0.0003,
    min_fee=5.0,
    stamp_duty=0.0005,
    slippage=0.0001,
    max_volume_ratio=0.1,
    matching_mode="close"
)

# 3. 运行多表回测 (主表自动推断，副表显式声明物理语义)
results = engine.run(
    strategy=multi_asset_strategy,
    data={
        "stock": pl.read_parquet("data/parquet/ashare.kline.1d/**/*.parquet"),
        "index": ts_table(pl.read_parquet("data/parquet/aindex.kline.1d/**/*.parquet")),
    },
)

# 4. 输出回测绩效与 Polars 交易日志
print(results.summary())
print(results.trade_logs)  # Polars DataFrame
```
