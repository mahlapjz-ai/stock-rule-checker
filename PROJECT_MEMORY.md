# 项目核心记忆
1. 架构：单文件 `规则一致性检测器.html`，部署于 GitHub Pages。
2. 核心铁律：严禁修改 `evaluateDayRules`、`computeScore`、`backtest` 这三个函数的逻辑，所有新增功能只能在外层包装。
3. 数据源：腾讯自选股公开接口。B/S标记为过去120天历史扫描，非实盘持仓标记。
4. 交互约束：所有变更必须通过本地 `regression_test.py` 的 6 项回归测试。
