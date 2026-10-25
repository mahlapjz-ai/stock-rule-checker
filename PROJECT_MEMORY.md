# 项目核心记忆

## 架构与铁律
1. 架构：单文件 `规则一致性检测器.html`，部署于 GitHub Pages。
2. 核心铁律：严禁修改 `evaluateDayRules`、`computeScore`、`backtest` 这三个函数的逻辑，所有新增功能只能在外层包装。
3. 数据源：腾讯自选股公开接口。B/S标记为过去120天历史扫描，非实盘持仓标记。
4. 交互约束：所有变更必须通过本地 `regression_test.py` 的 6 项回归测试。

## 图表标记与 Tooltip 渲染规范
5. B/S 标记防遮挡：`simpleAnnotation` 绘制时，主图须配 `gap: {top: 0.9, bottom: 0.25}`；标记偏移固定 `off: 0.08`（=(high-low)*0.08）；`zLevel: 10`，**不要**写 `fixedZLevel`（v9.6.0 无此字段）。
6. Tooltip 双端差异：移动端（<=768px）用 `showType: 'standard'`（文字固定左上、不随触点翻转，overlay 层标记绘制在文字上层永不遮挡）；桌面端用 `showType: 'rect'`（白底浮窗）。双端 `showRule` 统一 `'follow_cross'`。

## 开发规范（偏好）
7. 冗余文件清理：每次大任务收尾（合入 main、修复核心 bug、发版）后，须检查并清理本地冗余文件（临时 `test_*.py`、临时截图、废弃的中间 HTML 等）；清理前用 `git status` 确认不影响已跟踪的核心产出物（`规则一致性检测器.html`、`rules.json` 等）；保持工作区干净，只提交有意义的历史记录。
