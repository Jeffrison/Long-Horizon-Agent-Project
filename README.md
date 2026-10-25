# Long-Horizon Agent Evaluation & Post-training Data Pipeline

纯 Python 实现的 ReAct Agent，支持 Wikipedia 检索、Python 数学工具、可观测错误反馈，以及可审计的反思消融评测。项目重心是**公平评测、工具可靠性和轨迹数据工程**。

> 历史 README 的 8%→43%、8.33%→25% 来自存在重复运行/挑选偏差的实验，已撤回作为性能证据。原始代码和日志保留以便审计。本次真实复测结果以 reports 下的冻结实验为准，不预设反思一定有效。

## 实现能力

- **手写 ReAct 状态机**：协议解析 → 工具执行 → observation → 完成/预算耗尽。推理代码不接受标准答案。
- **等预算消融**：OFF/ON 都允许错误后继续；唯一干预是显式错误纠正指导语。每次请求，包括格式错误，都消耗预算。
- **轨迹可追溯**：保存每步实际输入 messages、模型输出、工具输入输出、错误类型、请求 id、token、finish reason、延迟。
- **评测治理**：问题级哈希划分、数据校验、预注册 manifest、三种子成对实验、逐条 checkpoint、失败保留、题目聚类 bootstrap。
- **保守判分**：QA answer EM/F1；数学整段表达式解析与受限 AST 数值计算，不再抽取第一数字或用子串判正确。
- **后训练数据工程**：仅从 dev 轨迹构造去重 SFT 和相同 prompt 的偏好候选，保留来源、结果标签和审核标志；可选 LoRA SFT 入口。
- **回归测试与 CI**：离线测试覆盖答案隔离、预算、错误恢复、判分、工具超时、数据划分和训练泄漏。

这不是完整的 RL/RLHF 训练系统；现阶段不能声称已经训练出效果更好的模型。Python 子进程有静态限制、环境清理和超时，但不是操作系统安全沙箱。

## 快速复现

要求 Python 3.10+。以下命令在仓库根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
# 在 .env 中配置自己的 LLM_API_KEY / LLM_BASE_URL，不要提交密钥
python -m eval.prepare_data
python -m pytest -q
```

先检查开发集，再冻结代码做完整复测：

```powershell
python -m eval.run_eval --split dev --limit 2 --seeds 11 --workers 2 --output outputs/my-smoke
python -m eval.run_eval --split test --seeds 11 29 47 --workers 4 --output outputs/my-test
python -m eval.report outputs/my-test
```

一个 question × seed × mode 只有一条轨迹。完整默认复测为 215 题 × 3 种子 × 2 模式 = **1290 条轨迹**，会调用收费模型 API。更改输出路径可启动独立预声明实验；禁止挑选表现更好的那次作为唯一结果。

中断后用完全相同参数加 `--resume`。已开始但没有完成记录的轨迹计为 interrupted failure，不能重新采样。不要同时向同一输出目录启动多个评测进程。

数据与配置在 manifest.json 中冻结；运行时不能更换代码、数据和 prompt。最终只保存了合法最终答案不代表答对，准确率由独立 scorer 判定。

## 结果与限制

本次正式复测报告完成后归档至 `reports/`，包含原始轨迹压缩归档、实验 manifest、环境快照、汇总和 Markdown 报告。运行中的实验可通过 `python -m eval.report <输出目录>` 查看，未完成的报告会显式标注进度。

- 原始数据：HotpotQA 风格问答 200 题、数学 60 题；数学原 ID 有 29 次重复，已在 validated 副本中修复。
- 冻结划分：dev 45 题，test 215 题；历史开发使用过这些题，不能宣称完全未见。
- QA 是实时 Wikipedia 检索，只报告答案 EM/F1，不是官方完整 HotpotQA distractor/supporting-facts 评测。
- seed 仅是向供应商发出的请求，不能保证远端推理确定性。重复种子的结果不会择优合并。
- 方法、提示词、工具与判分均已改变，不能把新旧数值相减后当作因果增益。

协议和误差分析见 [评测协议](docs/evaluation-protocol.md)；后训练路径、使用方法与边界见 [后训练文档](docs/post-training.md)。

## 后训练数据与可选 LoRA

```powershell
python -m eval.run_eval --split dev --seeds 11 29 47 --workers 4 --output outputs/my-dev
python -m post_training.export outputs/my-dev --output outputs/my-training-data
python -m post_training.train_sft --data outputs/my-training-data/sft.jsonl --dry-run
```

SFT 只学习最后一条 assistant completion；train/validation 按题目组划分。DPO 文件仅是需要审核的候选，最终答对不能证明每步动作优质。可选 GPU 训练依赖与 LoRA 命令见后训练文档。API rollout、数据导出、dry-run 与模型训练是不同完成状态。

## 目录

```text
agent/          模型 HTTP 客户端、协议解析、ReAct 状态机
tools/          typed result、受限 Python 子进程、Wikipedia 缓存
eval/           数据审计、哈希划分、成对评测、统计报告
post_training/  dev-only SFT / 偏好候选导出、可选 LoRA 入口
tests/          无需 API Key 的回归测试
data/           原数据与 validated 副本
logs/           历史日志（有偏，仅用于审计）
reports/        本次冻结实验与原始轨迹归档
docs/           协议、后训练说明、项目升级与简历材料
```

原始版本保留在 `codex/archive-original-d95783a`；本次开发分支为 `codex/reproducible-agent-eval`。
