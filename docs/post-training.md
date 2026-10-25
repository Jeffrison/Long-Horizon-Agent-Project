# 从 Agent 轨迹到后训练数据

已实现：按题目划分 → 开发集 rollout → 结果判分 → 去重、过滤与审计 → SFT 样本 / DPO 候选 → 可选 LoRA SFT 入口。

## 数据构造

```powershell
python -m eval.run_eval --split dev --seeds 11 29 47 --workers 4 --output outputs/dev-rollouts
python -m post_training.export outputs/dev-rollouts --output outputs/training-data
python -m post_training.train_sft --data outputs/training-data/sft.jsonl --dry-run
```

- 只有 dev 题目能导出；重新计算题目哈希和 split，test 数据不会被接受。
- SFT 从最终答对且协议完成的轨迹中导出成功工具调用/最终回答的单步监督。保留该步真实输入 messages；错误步骤和被截断的输出排除。
- 监督仍可能包含错误推理；“最后答对”不保证每个步骤都对。它是 outcome-filtered self-generated data，不是专家数据。
- 偏好候选只比较**完整输入消息完全相同**的不同 assistant 输出，不能简单拼接不同 prompt 的整条轨迹。
- chosen 来自最终答对轨迹，rejected 来自最终答错但正常结束轨迹；不把 API 故障或截断当作负样本。
- 早期动作与最终成败的因果关系不确定，因此输出文件命名为 preference_candidates；metadata.requires_review=true。未审核不能宣称高质量偏好数据或已训练 DPO。
- 保留 source revision、episode key、question key 与输入 manifest 哈希，支持追溯。
- 不使用旧的人工挑选日志训练。

## 可选 LoRA SFT

```powershell
# 在单独的 CUDA 环境中执行
python -m pip install -r requirements-train.txt
python -m post_training.train_sft --data outputs/training-data/sft.jsonl --output outputs/lora-sft --model Qwen/Qwen2.5-7B-Instruct --model-revision <固定的HF提交哈希>
```

入口按题目组在 dev 内再分 train/validation，防止同题不同轨迹跨越这两个集合。对最后一条 completion 计算 loss；工具 Observation 和历史上下文不作为当前步的预测目标。训练前检查完整序列长度，禁止静默截去目标答案。LoRA r=16 / alpha=32 / all-linear；结果和 resolved model revision 写入 data_manifest。

实现参考 [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) 和 [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora)。可选训练依赖未在本机完整安装，7B GPU 训练与权重效果验证尚未执行；dry-run 仅验证数据，不是训练。实际使用应在 GPU 环境锁定依赖版本与模型提交，再验证 loss masking、显存和训练收敛。

## 接下来可以验证的研究假设

1. 先将“错误反馈提示”与“模型生成的反思步骤”拆开做同预算消融，避免把提示收益统称为反思收益。
2. 扩充新的、未参与调试的题目来源，冻结并独立保管最终测试集。
3. 固定基座、数据与推理预算，对比 base / SFT-LoRA / 审核后 DPO；用未见题目衡量泛化。
4. 对同 prompt 多采样构造偏好时，保留所有采样，比较 pass@1 与单独标注的 best-of-N，避免重演挑选偏差。
5. 在当前数据规模不足时，先做错误归因与数据质量实验，不把小样本数值包装成模型训练突破。
