# ReAct Agent、评测与后训练实验

在原有多跳问答 / Python 工具 Agent 上完善的学习项目：
Agent 执行 → 可追溯评测 → 数据筛选 → 小模型 LoRA SFT / DPO → 独立评测。

## 已实现
- direct、react、react_reflect 模式，共享模型/工具调用上限；格式重试也消耗预算。
- Wikipedia 检索和可选 Python 数学执行，结构化返回结果；有限纠错与滑动窗口历史。
- Agent 接口不接收标准答案，只在运行结束后评分。
- 答案级 HotpotQA EM/F1、逐步上下文、错误分类、token 与耗时记录。
- 问题哈希切分，数据与源码指纹；可追溯 SFT/偏好数据导出。
- 小模型 LoRA SFT/DPO 训练入口与本地 adapter 评测，支持 completion-only SFT。
- 25 项离线回归测试和无需密钥的端到端演示。

## 实验口径
- 旧 README 的 8%/43% 为宽松自定义匹配。重算旧 QA：OFF 的 EM=4.0%、F1=6.22%；ON 的 EM=29.5%、F1=38.21%。
- 作者补充：历史 ON 组经过额外采样和按正确性筛选，缺少全部失败尝试。上述分数仅描述已留存记录，严格重评分也不能恢复无偏 pass@1/pass@k，更不能归因于反思。当前不评 supporting facts / joint 分数。
- 旧数学轨迹用标准答案提前停止，不用于效果宣传或训练导出。
- 200 个 QA 问题以前均评测过，当前划分是回溯划分，不是全新盲测。
- 旧记录只导出答案级训练数据，不代表训练了工具选择或 Agentic RL。
- 成功示范可用于训练，但来源属于按结果筛选的数据。规则标签为 silver labels，可能错拒同义答案，偏好对需要人工复核。
- Python 子进程不是安全沙箱，数学执行需显式 --allow-python，只用于本人可控实验。
- 实训和效果状态看 reports/validation.json；有训练脚本不等于训练已经成功。

## 离线验证
Python 3.10+，在项目根目录运行：

    python -m unittest discover -s tests -v
    python -m examples.offline_demo

演示是虚构实体与固定响应，只验证工具闭环，不是模型 benchmark。

## 历史重评分

    python -m eval.rescore logs/Eval_QA_Reflect_OFF_20260504_181855.json logs/Eval_QA_Reflect_ON_20260502_153421.json

原 logs 不修改，结果写入 outputs/reports。旧 ON 组仅统计成功筛选后留存的记录，不能恢复完整采样分母。

## 固定预算的反思与采样对照

新增 python -m eval.fair_compare --backend fixture，演示全量留痕协议；fixture 不是模型效果证据。
真实实验比较单次 ReAct、带错误反馈的 ReAct 和独立重采样。预先保存计划、固定预算，保留全部尝试，分别报告多数投票结果和 oracle 至少一次正确的上界。
详细说明见 docs/成功样本筛选与反思对照.md。

## 构建数据

    python -m posttrain.build_data --logs logs/Eval_QA_Reflect_OFF_20260504_181855.json logs/Eval_QA_Reflect_ON_20260502_153421.json

新默认输出为 outputs/posttrain_answer_v2，包含采集与筛选来源说明；原 outputs/posttrain_answer 保留为已完成实训的数据快照。
按问题哈希、种子 42 约 80/10/10 划分：

| 数据 | train | validation | test |
|---|---:|---:|---:|
| 答案级 SFT | 49 | 6 | 7 |
| 答案级 DPO | 15 | 0 | 2 |

偏好验证集为空，不能从测试集复制填充。所有规则标签初始为 pending。
明确接受规则标签的教学实验可运行：

    python -m posttrain.train --method sft --dry-run --accept-silver-labels
    python -m posttrain.train --method dpo --dry-run --accept-silver-labels

新版 Agent 日志可导出精确上下文决策数据：

    python -m posttrain.build_data --logs outputs/eval/新版QA日志.json --kind decision --output outputs/posttrain_decision

决策 SFT 用真实 messages 与一次 assistant 输出；不监督工具返回。
决策 DPO 只在相同前缀下配对正确/错误最终决策，没有合格偏好对就导出零条。

## 8GB GPU 训练
小模型入门配方：0.5B、LoRA rank=8、max-length=512、micro-batch=1、
梯度累积=8、梯度检查点；实际显存随环境和序列变化。
依赖范围见 posttrain/requirements.txt，实际验证版本见 posttrain/requirements-tested.txt。当前机器建立了项目内 .venv-train，复用现有 CUDA 包。
先用 python -m scripts.download_model 下载固定版本并逐文件校验哈希（默认 Qwen 官方 ModelScope 地址）。

    .\.venv-train\Scripts\python.exe -m posttrain.train --method sft --model outputs/models/qwen2.5-0.5b-instruct --accept-silver-labels --max-steps 20 --output outputs/sft_adapter --export-merged
    .\.venv-train\Scripts\python.exe -m posttrain.train --method dpo --model outputs/sft_adapter/merged --accept-silver-labels --max-steps 5 --output outputs/dpo_adapter

SFT 保存 adapter，--export-merged 另存合并权重。DPO 使用 SFT 合并权重作为参考基座，
才是 SFT→DPO；使用原始基座则是独立 DPO 实验。
训练报告保存软件版本、数据哈希、损失与显存。损失下降不等于任务准确率提升。

## 评测
API 方式需本地 .env 中的 LLM_BASE_URL / LLM_API_KEY：

    python -m eval.run_eval --mode all --split test --max-samples 10

会真实调用模型与 Wikipedia；3 模式 × 10 题 × 8 次上限，最多 240 次模型调用。
调用上限相同不等于实际 token 成本相同，因此分别报告调用数、token 和延迟。

本地基座与 adapter 对照：

    .\.venv-train\Scripts\python.exe -m eval.run_eval --backend local --model outputs/models/qwen2.5-0.5b-instruct --mode direct --split test --max-samples 10 --temperature 0 --max-new-tokens 64 --output outputs/base_eval
    .\.venv-train\Scripts\python.exe -m eval.run_eval --backend local --model outputs/models/qwen2.5-0.5b-instruct --adapter outputs/sft_adapter --mode direct --split test --max-samples 10 --temperature 0 --max-new-tokens 64 --output outputs/sft_eval

本轮已实际完成 SFT 20 步和 DPO 5 步，以及 10 题同题直接回答测试：基座 EM/F1=30%/30%，SFT=20%/20%，DPO=20%/20%。没有观察到提升。首次诊断曾错误要求 direct 回答使用 ReAct 前缀，三组均格式失败；已保留记录并统一改用直接回答协议重测。reports/validation.json 的 latest_evaluation 是最新同协议结果。
这只比较直接回答。Agent 提升还需相同工具、检索环境与预算的 react 测试。
DPO adapter 的基座必须与训练使用的基座一致。

## 目录与版本
agent/ 是运行时；tools/ 是工具；eval/ 是评测；posttrain/ 是训练数据和训练入口。
tests/ 是回归；docs/ 是学习路线和简历；reports/ 是小型审计报告。
outputs/ 保存生成数据、模型和大日志，Git 忽略，可按命令重建；.env 不提交。
本次升级分支 improve/agent-posttraining，原代码保留在 main。
查看差异：git diff main...HEAD；查看阶段：git log --oneline。

## 参考
- HotpotQA：https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
- TRL SFT：https://huggingface.co/docs/trl/sft_trainer
- TRL DPO：https://huggingface.co/docs/trl/dpo_trainer
- Qwen：https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
