# 银行数据质量处理与分类智能体

本项目面向人工注入脏数据的银行营销数据集，完成数据质量画像、清洗规划、无泄漏清洗、质量复检、失败重试和 `y` 二分类。支持规则规划器与 OpenAI 兼容的大模型规划器；大模型只能生成计划，实际数据处理始终由确定性工具执行。

## 已实现能力

- 检测缺失、完全重复、数值格式、类别空格/大小写、IQR 异常、年龄业务异常、恒定字段和标签分布。
- 生成结构化清洗计划，并进行顶层结构、操作白名单、字段范围和业务安全校验。
- 执行“计划 → 校验 → 清洗 → 复检 → 带反馈重规划”的闭环。
- 大模型或计划连续失败时可显式回退到规则计划，日志标记 `fallback_used=true`。
- 去除完全重复记录后再进行分层训练/测试划分。
- 解析 `43岁`、`EUR -583` 等格式，统一类别空格与大小写。
- 保留原始 `unknown` 业务类别；空白、`?` 等才按缺失处理。
- 将年龄不在 `[18, 100]` 的值设为无效，使用训练集年龄中位数填充，并保留缺失/异常指示列。
- 负余额不删除，极端值不自动截断；恒定字段只有在原始报告和训练集均支持时才删除。
- 训练类别不平衡逻辑回归，导出清洗数据、预测概率、质量报告、模型工件和审计日志。

## 项目结构

```text
.
├── run_pipeline.py          # 固定规则基线入口
├── run_agent.py             # 智能体入口
├── verify_run.py            # 输出一致性与模型复现核验
├── requirements.txt
├── .env.example             # 大模型配置模板，不含真实密钥
├── src/
│   ├── constants.py         # 字段与标签约定
│   ├── data.py              # 读取、缺失识别、格式解析、去重
│   ├── quality.py           # 数据质量画像
│   ├── planning.py          # 规则型清洗计划
│   ├── plan_schema.py       # 计划结构、业务安全校验与复检规则
│   ├── planner.py           # 规则/大模型规划器及 API 调用记录
│   ├── cleaning.py          # 计划驱动、训练集内拟合的清洗器
│   ├── agent.py             # 校验、执行、复检、重试和回退
│   ├── modeling.py          # 分类模型与评估
│   ├── pipeline.py          # 固定基线编排
│   └── agent_pipeline.py    # 智能体结果与审计日志导出
├── tests/
│   ├── test_core.py
│   └── test_agent.py
├── outputs/                 # 固定基线结果
└── agent_outputs/           # 智能体运行结果
```

## 1. 运行固定基线

建议使用 Python 3.10 或更高版本。

```bash
pip install -r requirements.txt
python run_pipeline.py --input "银行数据集（已添加问题）.csv" --output-dir outputs
```

## 2. 运行智能体：规则模式

规则模式不需要网络或 API 密钥，用于测试完整闭环和离线演示：

```bash
python run_agent.py --planner rule --output-dir agent_outputs
```

## 3. 运行智能体：大模型模式

支持兼容 `POST /chat/completions` 的 API。复制 `.env.example` 中的变量名，在终端设置真实配置。PowerShell 示例：

```powershell
$env:LLM_API_KEY="你的密钥"
$env:LLM_BASE_URL="https://你的服务地址/v1"
$env:LLM_MODEL="你的模型名称"
$env:LLM_INPUT_COST_PER_1M="每百万输入token的美元单价"
$env:LLM_OUTPUT_COST_PER_1M="每百万输出token的美元单价"
python run_agent.py --planner llm --max-attempts 3
```

价格变量是可选的；不配置时仍记录 token 用量，但估算费用为 `null`。程序不会把 API 密钥写入日志。

默认在大模型调用、计划校验或执行复检连续失败后回退到规则计划。若实验要求只接受大模型成功结果：

```bash
python run_agent.py --planner llm --max-attempts 3 --no-fallback
```

遇到 HTTP 429 或 5xx 时会按 `Retry-After` 或指数退避等待，不进行紧密循环重试。

## 测试与核验

```bash
python -m unittest discover -s tests -v
python verify_run.py --output-dir outputs
python verify_run.py --output-dir agent_outputs
```

当前 12 项测试覆盖格式恢复、缺失识别、标签保护、训练集中位数、计划安全校验、非法计划重试、复检后重试和规则回退。

## 输出文件

| 文件 | 内容 |
|---|---|
| `quality_report_before.json` | 原始数据质量画像 |
| `cleaning_plan.json` | 最终接受的清洗计划及原因 |
| `cleaner_metadata.json` | 训练集填充值、删除字段和建模特征 |
| `cleaned_data.csv` | 去重并清洗后的全部数据，含原始行号 |
| `cleaned_train.csv` / `cleaned_test.csv` | 分层划分后的清洗数据 |
| `quality_report_after.json` | 清洗后的质量复检 |
| `agent_attempts.json` | 每轮计划、校验错误、执行错误、残留问题与回退状态 |
| `llm_call_log.json` | 提示词、模型参数、原始响应、token 用量、费用和 API 错误 |
| `metrics.json` | PR-AUC、ROC-AUC、F1、召回率、混淆矩阵及智能体状态 |
| `test_predictions.csv` | 测试集真实标签、预测标签和正类概率 |
| `agent_artifacts.joblib` | 智能体模式清洗器、计划和分类流水线 |
| `baseline_artifacts.joblib` | 固定基线清洗器和分类流水线 |
| `run_log.json` | 参数、环境、输入哈希、调用次数、总费用与运行状态 |

## 当前结果

使用 20% 分层测试集、随机种子 42、类别平衡逻辑回归：

- 原始记录 21,050 条，去除 50 条完全重复记录。
- 训练集 16,800 条，测试集 4,200 条。
- PR-AUC：0.5095；ROC-AUC：0.9538。
- 正类精确率：0.3452；召回率：0.9109；F1：0.5007。
- 混淆矩阵：TN=3649，FP=349，FN=18，TP=184。
- 规则智能体一次通过复检，未使用回退，指标与固定基线一致。

这里以 PR-AUC、召回率和 F1 为主，不因类别不平衡而只看准确率。真实大模型模式需要用户提供 API 配置后另行验证，不能用规则模式结果代替。

## 防止评估失真的约定

1. 完全重复记录先删除，再划分训练集和测试集。
2. 清洗器仅在训练集上 `fit`，测试集只调用 `transform`。
3. 标签 `y` 只校验，不作为清洗对象。
4. 大模型不得直接改写数据文件，只能提交通过校验的白名单计划。
5. `duration` 是通话时长；若业务场景是通话前预测，应在后续实验中排除。
6. 清洗后质量报告用于验证处理规则，不代表可以再用全部数据重新拟合清洗器。
