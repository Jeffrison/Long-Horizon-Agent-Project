# 从 Agent 轨迹到 LoRA SFT

已经跑通：按题目划分 → 开发集 rollout → 独立判分 → 过滤与去重 → SFT / 偏好候选 → 实际 0.5B LoRA 训练 → loss-mask 审计 → 基座/适配器验证。

训练证据与适用范围见 [0.5B LoRA 验证](lora-pilot.md)。7B 模型只进行了 API Agent 评测；没有 7B 本地训练，也没有 DPO/RLHF/GRPO 效果结论。

## 数据构造

```powershell
python -m eval.run_eval --split dev --seeds 11 29 47 --workers 4 --api-interval 1 --output outputs/my-dev
python -m post_training.export outputs/my-dev --output outputs/my-training-data
python -m post_training.train_sft --data outputs/my-training-data/sft.jsonl --dry-run
```

- 仅 dev 题目可导出；重新计算题目哈希和 split，不接受 test 轨迹。
- SFT 导出最终答对且正常完成轨迹中的成功工具调用和最终回复，保留该步真实输入 messages；排除错误步骤及截断输出。
- 最后答对不保证每步推理优质；这是 outcome-filtered self-generated data，不是专家轨迹。可记录为教师 rollout 数据构建。
- 偏好候选只比较**完整输入 messages 完全相同**的不同 assistant 输出，不拼接不同上下文的轨迹。
- chosen 来自最终答对轨迹；rejected 来自答错但正常结束轨迹。API 故障、截断和预算耗尽不当作负样本。
- 早期动作与最终成败的因果关系不明确，DPO 候选都带 requires_review=true。未经审核不能声称高质量偏好数据。
- metadata 包含 source revision、episode key、question key；导出 manifest 记录输入 manifest 哈希。
- 历史人工挑选日志不用于本次训练。

本次真实导出：QA 78 条 SFT / 2 组偏好候选，数学 14 条 SFT / 1 组偏好候选。实际 LoRA pilot 只使用 QA 的 78 条；没有把数学和 QA 训练数据混成一个效果结论。

## 训练实现

```powershell
# 单独 CUDA 环境；实际验证版本见 requirements-train-tested.txt
python -m pip install -r requirements-train.txt
python -m post_training.train_sft --data <sft.jsonl> --output outputs/my-lora --model Qwen/Qwen2.5-0.5B-Instruct --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 --max-steps 25
```

同题的不同轨迹必须放在同一 train/validation 分组。每个样本只学习最后一条 assistant completion；之前的 assistant、用户输入和工具 observation 全部作为上下文，loss=-100。训练前检查完整序列长度，超过预算则拒绝训练，不静默丢弃目标。

本次实测：63 条/15 题训练，15 条/3 题验证；25 steps；LoRA r=16 / alpha=32 / dropout=0.05 / all-linear；bf16。适配器权重、本地路径与 SHA256 分开保留，Git 归档保存训练与验证证据。

实现参考 [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) 与 [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora)。

## 审计与验证

```powershell
python -m post_training.audit_mask --data <sft.jsonl> --model <base> --revision <hash> --output outputs/mask.json
python -m post_training.evaluate_sft --data <sft.jsonl> --adapter <adapter目录> --model <base> --model-revision <hash> --output outputs/validation.json
python -m agent.run --backend local --model <base> --revision <hash> --adapter <adapter目录> --query <问题>
```

audit_mask 调用真实 TRL 预处理和 collator 检查实际 labels。evaluate_sft 用相同 prompt 前缀比较基座/LoRA 的 token 加权 completion NLL，验证题从训练开发集中按题目隔离，不使用 test。NLL 不是答题准确率，3 道验证题也不足以证明泛化收益。

## 接下来的研究方向

1. 使用新来源、未参与调试的题目建立独立测试集，检查基座预训练数据污染。
2. 固定基座、数据和推理预算，对比 base/SFT-LoRA；补实际 Agent 答案指标，而非只看训练 loss。
3. 人工审核同 prompt 偏好，扩充数量后再做 DPO，保持独立验证与测试集。
4. 将显式错误反馈与模型生成的反思步骤拆开消融；当前 ON/OFF 是 observable-error feedback，自我纠错提示并不等同语义反思算法。
5. 多采样生成训练数据时保留采样总数；评测 pass@1 与 best-of-N 分开，禁止重演挑选偏差。
