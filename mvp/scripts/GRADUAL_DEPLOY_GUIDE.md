# 灰度实盘（小资金）部署与运行流程指南

> 目标：以总资金的 0.5% - 1% 先行上线，逐步放量，确保稳定与安全。本文档覆盖：上线前检查项、上线步骤、回退方案，以及可执行脚本说明。

## 一、上线前检查项（Preflight）

强烈建议在每次灰度上线前执行 `preflight` 自检：

- OKX API 健康：请求公共时间接口，要求 HTTP 200 且延迟可接受（例如 < 300ms）。
- 数据延迟：数据库中 `trades` 表的最新时间与当前 UTC 的差值应小于阈值（默认 5s）。
- 回测阈值：策略近期回测需要达到最低门槛，例如：
  - 夏普率 Sharpe >= 0.2（示例值，按需调整）
  - 胜率 Winrate >= 50%
  - 最大回撤 Max Drawdown <= 5%

执行命令（Windows PowerShell）：

```
py scripts/gradual_deploy.py preflight --http-admin-url http://127.0.0.1:8001 \
  --backtest-summary data/backtest_summary.json --min-sharpe 0.2 --min-winrate 0.5 --max-drawdown 0.05
```

说明：
- 若未启用 HTTP 管理接口，可省略 `--http-admin-url`，仅进行 API 与数据、回测检测。
- 若数据库未配置，可添加 `--skip-db-check` 暂时跳过数据延迟检查。
- 若回测结果文件名不同，请调整 `--backtest-summary` 路径。

## 二、上线步骤（逐步放量）

1. 启动系统主程序（另一个终端窗口）：
   - 建议首先在 mock 模式启动，确认链路通。
   - 关键环境变量（示例）：
     - EXEC_MODE=mock
     - EXEC_HTTP_ADMIN_ENABLED=true
     - EXEC_HTTP_HOST=127.0.0.1
     - EXEC_HTTP_PORT=8001
     - EXEC_CONTROL_FILE=data/control/mode.json
     - AUDIT_TO_FILE=true，AUDIT_TO_DB=false（如数据库未就绪）
     - DATABASE_URL=postgresql://user:pass@localhost:5432/yourdb（若写 DB）

2. 执行上线前自检（必须通过）：

```
py scripts/gradual_deploy.py preflight --http-admin-url http://127.0.0.1:8001 \
  --backtest-summary data/backtest_summary.json --min-sharpe 0.2 --min-winrate 0.5 --max-drawdown 0.05
```

3. 切换到 mock（灰度演练）：

```
py scripts/gradual_deploy.py switch --to mock --http-admin-url http://127.0.0.1:8001
```

- 观察 10-30 分钟：
  - data/audit/ 下 JSONL 按天滚动是否正常写入
  - data/risk_events.csv 是否出现异常事件
  - 策略是否有合理信号频率与风控拦截

4. 切换到 real（仅用 0.5%-1% 的额度/仓位）：

```
py scripts/gradual_deploy.py switch --to real --http-admin-url http://127.0.0.1:8001
```

- 请在配置中确保：
  - 单笔名义资金/下单数量受限（RiskManager 与策略参数）
  - 单笔/日内最大亏损阈值较严（SINGLE_TRADE_MAX_LOSS_PCT / DAILY_MAX_LOSS_PCT）
  - 若出现连续风控/审计异常，及时回退到 `paused`

5. 观察期（至少 1-2 小时，小资金）：
   - 实盘回报、成交滑点、下单失败率
   - 审计日志（order_intent / order_result / risk_block / risk_event）
   - 如表现正常，逐步提高额度（谨慎、分步）

## 三、回退方案（故障/异常）

当出现以下任一情况，立即“紧急回退”：
- 策略出现明显异常（爆量、异常价格、频繁被风控拦截）
- 交易所 API 不稳定或错误率升高
- 数据延迟超出阈值，或行情明显滞后

回退操作（两种方式，推荐优先 HTTP 管理接口）：

1) HTTP 管理接口停止：
```
py scripts/gradual_deploy.py pause --http-admin-url http://127.0.0.1:8001
```

2) 写控制文件（适用于未启用/失联的 HTTP 管理接口）：
```
py scripts/gradual_deploy.py pause --control-file data/control/mode.json
```

系统会切换到 `paused`，所有下单会被拒绝并记录审计。排查完成且确认安全后，再切回 `mock` 演练，最终恢复到 `real`。

## 四、脚本参数说明

- preflight
  - --http-admin-url：HTTP 管理端根地址（如启用）
  - --backtest-summary：回测结果 JSON 路径
  - --min-sharpe / --min-winrate / --max-drawdown：回测阈值
  - --data-latency-threshold：数据延迟阈值（秒）
  - --skip-backtest-check / --skip-db-check：跳过指定检查项
  - --control-file：控制文件路径（未启用 HTTP 管理接口时使用）

- switch
  - --to：目标模式 mock|real|paused
  - --http-admin-url：HTTP 管理端地址
  - --control-file：控制文件路径

- pause
  - --http-admin-url / --control-file：同上

## 五、依赖与运行

- 安装依赖（若缺失）：
```
py -m pip install -r requirements.txt
```
- 如果仅运行脚本且未使用项目其它模块，最少依赖为：
```
py -m pip install requests
```

- 运行：
```
py scripts/gradual_deploy.py --help
```

## 六、最佳实践与注意事项

- 永远从 mock 演练开始，确认链路、风控与审计正常后再 real。
- 严格限制初期实盘的下单金额与风控阈值，必要时启用更严格的熔断。
- 关注 data/audit/*.jsonl 与数据库中的审计/风控记录，复盘问题时有帮助。
- 不要在代码中硬编码 API Key，统一通过环境变量或 config.yaml。
- 实盘密钥请使用只读/限额账户，或子账户限额，降低风险。
- 保持回测文件更新（例如每日或每次参数更新后重新评估）。