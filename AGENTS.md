# AGENTS.md - CarrotQuant Engine AI Agent 工程与架构指南

## 1. 项目定位与核心设计
`CarrotQuant Engine` (`carrotquant-engine`) 是基于 Python/Numba 的全市场 (A股/美股/期货) 事件驱动与向量化量化回测引擎。

核心设计：
- **内存与计算优化**：采用 2D NumPy C-Contiguous 内存布局与 SoA 预分配数组，核心撮合使用 Numba `@njit(nogil=True)`。
- **严禁黑盒误判与模糊猜测（红线）**：量化回测绝不能有黑盒误判和模糊猜测。副表/自定义数据供给严禁隐式模棱两可的盲目推测，必须通过显式声明 (`cq.engine.ts_table`, `cq.engine.event_table`, `cq.engine.static_table` 或 `(df, "ts"|"event"|"static")` 元组或预构建容器) 明确物理语义；未显式声明且存在歧义时必须显式抛错提示。
- **Duck Typing 数据协议与统一 Dict 供给**：`engine.run(data=...)` 原生支持标准 `dict` 多表供给（主行情表、副行情表、多因子特征表、离散事件表、静态属性表），支持 Query/Reader 鸭子对象 (`.to_df()` / `.read()`)、Polars `LazyFrame` (`.collect()`)、`DataFrame` 及分块流式生成器；`cq.engine` 与 `cq.data` 保持物理隔离与零硬依赖。
- **策略接口与数据防窥**：提供 `@strategy` 装饰器，运行时通过 `[:t+1]` 物理边界切片严格防止未来函数污染；支持 `ctx.get("index").close` 层级访问与 `ctx.get("dragon_tiger")` 稀疏事件查询。
- **统一表级层级访问契约**：策略上下文统一遵循 `ctx.get("表名").列名` 访问模式：
  - TS 表（行情/多因子）：`ctx.get("index").close` 获取当前步向量 $(N,)$，`ctx.get("valuation").pe_ttm` 获取特征向量，`ctx.get("valuation").get_history("pe_ttm")` 获取历史矩阵切片；
  - Event 表（离散事件）：`ctx.get("dragon_tiger")` 获取当前步 `EventSnapshot`，未发生事件标的返回 `None`；
  - Static 表（静态属性）：`ctx.get("concept")` 获取 `StaticAttributeContainer`，支持一对多列表聚合，未分类返回 `None`；
  - 主表快捷方式：`ctx.close` 作为主行情表 `ctx.get("stock").close` 的高频便捷别名。
- **单趟极速矩阵构建 (MatrixBuilder)**：单趟完成坐标规整与内存对齐，主 TS 行情表严格校验 OHLC 全量字段，缺失即报 `ValueError`；未提供 `volume`/`amount` 时保持纯净 `None`。
- **Master Clock 时空对齐**：副 TS 表按主时钟 Left Join 内存对齐，超出时间步截断，缺失时间步填充 `NaN`，所有矩阵第二维度严格对齐至 $N$。
- **多空撮合机制与流式预热**：`buy/sell` 支持多空双向交易；支持滑点、印花税、最低佣金、`max_volume_ratio` 盘口上限与 `warmup_steps` 流式分块预热切片。
- **价格分层**：`data.close` 为原始成交价（用于资金交割与持仓成本），`data.adj.close` / `ctx.adj.close` 为动态后复权价（用于计算指标）。

## 2. 代码分层与架构规范
- `cq.engine.feed`:
  - `matrix_builder.py`: 单趟极速矩阵构建器、`TimeSeriesTable`、OHLC 强制校验、`MarketData` 容器、`SparseEventContainer` 与 `StaticAttributeContainer`。
  - `feed_loader.py`: Duck Typing 协议解构、显式表类型包装器 (`ts_table`, `event_table`, `static_table`)、Master Clock 对齐与多表装载 `load_feed` / `stream_feed`。
  - `column_loader.py`: Parquet/CSV 列式读取与 Hive 分区装载。
  - `chunk_streamer.py`: 内存 Window 流式分块器。
- `cq.engine`: JIT 撮合内核 (`matching.py`)、`MatchingMode` 枚举/参数解析与 SoA 状态管理 (`state.py`)。
- `cq.engine.strategy`: `@strategy` 装饰器与 `BarContext` / `TableContext` 上下文切片 (`ctx.close`, `ctx.get('index').close_history`, `ctx.get('valuation').pe_ttm`)。
- `cq.engine.indicators`: Numba 兼容的递推指标算子。
- `cq.engine.analytics`: 回测结果汇总、交易日志与 Polars 绩效度量。
- `cq.engine.utils`:
  - `time_utils.py`: 单点权威时间解析 (`parse_date_to_ms`)、格式化 (`ts_to_iso_str`) 与标准 `timestamp` (Int64) 提取。

## 3. 执行模式
- **统一入口 API (`engine.run`)**:
  - **多表/单表 Dict 与显式类型模式**:
    ```python
    engine.run(
        strategy=my_strat,
        data={
            "stock": stock_df,
            "index": cq.engine.ts_table(index_df),
            "valuation": cq.engine.ts_table(val_df),
            "dragon_tiger": cq.engine.event_table(dt_df),
            "concept": cq.engine.static_table(concept_df),
        }
    )
    ```
  - **向量化矩阵模式**: `engine.run(signals=signals, amounts=amounts, data=data)`
  - **分块流式预热模式**: `engine.run(strategy=my_strat, data=stream_feed(...), warmup_steps=10)`

## 4. 开发与测试准则
- 所有内核算子必须经过 Numba `@njit(nogil=True)` 编译验证。
- 测试套件必须 100% 独立于外部磁盘物理 `data/` 路径，采用内存与 `tmp_path` 构造 Mock 数据测试。
- 必须通过 `test_anti_lookahead.py` 防未来函数 Chaos 混沌注入验证。
- 保持单元与集成测试覆盖率高于 85%。
