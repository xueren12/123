# 加密货币日内高频量化交易系统（Python 项目骨架）

本项目旨在搭建一个面向 BTC/ETH 等主流币的秒级交易系统，支持数据采集、策略、交易执行、AI 辅助分析、监控告警及回测模块。该仓库当前为最小可运行骨架，后续可按模块逐步扩展。

## 项目目标
- 面向日内高频（秒级）行情与交易
- 模块化设计，模块可独立运行与测试
- 数据源：OKX WebSocket（trades、books5）
- 数据落库：PostgreSQL（TimescaleDB 扩展）
- 策略：初始提供盘口深度驱动策略
- 执行：支持模拟盘与实盘、限价/市价/止盈止损
- AI：融合行情、链上与舆情，生成建议与风险提示
- 监控：终端 + Telegram/Slack 告警
- 回测：秒/分钟级别回测与指标评估

## 目录结构
```
.
├── data/          # 数据存储（原始与导出）
├── strategies/    # 策略模块
├── models/        # AI 模型与特征处理
├── executor/      # 交易执行模块
├── utils/         # 工具函数
├── main.py        # 程序入口（演示）
└── requirements.txt
```

## 技术栈
- 语言：Python 3.10+
- 数据处理：pandas / numpy
- 异步与网络：websockets / httpx
- 数据库：PostgreSQL + TimescaleDB，psycopg3 + SQLAlchemy
- 监控与告警：loguru / python-telegram-bot / slack_sdk
- 回测与可视化：numba / scipy / matplotlib
- AI：transformers / scikit-learn（可选）

## 模块功能规划
1. 数据采集
   - 通过 OKX WebSocket 订阅 trades 与 books5
   - 实时写入 PostgreSQL（TimescaleDB）
   - 支持配置交易对与数据库连接
2. 数据库工具
   - 提供写入、查询、时间窗口抽取接口，输出 pandas.DataFrame
3. 策略模块
   - 初版盘口深度驱动（买盘>卖盘开多，反之开空），每秒更新信号
   - 统一信号接口：`{ts, symbol, side, price, size, reason}`
4. 交易执行
   - 支持模拟盘与实盘切换
   - 限价/市价/止损/止盈，记录交易日志
5. AI 模块
   - 接入行情、链上大额转账、社交舆情，事件触发式推理，降低 token 消耗
6. 监控模块
   - 实时展示行情、策略信号、交易状态
   - 价格跳变、交易失败、API 断开等异常触发告警（终端/Telegram/Slack）
7. 回测模块
   - 基于历史 trades 与 orderbook，支持秒/分钟级回测
   - 输出收益曲线、夏普、最大回撤、胜率等指标，生成 CSV/图表报告

## 快速开始
1. 克隆仓库并安装依赖
   ```bash
   pip install -r requirements.txt
   ```
2. 运行主程序（当前仅打印）
   ```bash
   python main.py
   ```
3. 创建 `.env` 与 `config.yml`（后续模块将读取）
   - `.env` 示例：数据库连接、交易所 API Key、Telegram Bot Token 等
   - `config.yml` 示例：订阅交易对、回测区间、策略参数、告警配置等

## 后续开发路线
- [ ] 数据库工具与 TimescaleDB 表结构初始化脚本
- [ ] OKX 实时数据采集器（trades & books5）
- [ ] 策略与执行模块的对接与回测框架
- [ ] 监控告警与消息路由
- [ ] AI 事件触发式推理与缓存

如需我继续为你补齐各模块代码与配置，请告诉我你的数据库与交易所环境（PostgreSQL/TimescaleDB 地址、OKX API Key 是否准备好等）。