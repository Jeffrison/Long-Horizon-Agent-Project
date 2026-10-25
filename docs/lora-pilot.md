# 0.5B LoRA SFT 验证（2026-10-25）

这是开发集上的小规模训练链路验证，与 7B API 的反思评测分开记录。

- 基座：Qwen/Qwen2.5-0.5B-Instruct，提交 7ae557604adf67be50417f59c2c2f167def9a775。
- GPU：RTX 4060 Laptop，8GB；bf16，LoRA r=16 / alpha=32 / dropout=0.05 / all-linear。
- 数据：从 QA 开发集首次运行的成功正确轨迹中导出 78 条单步 SFT。63 条/15 题训练，15 条/3 题验证，题目互不重叠。
- 优化：25 steps，batch=1，gradient accumulation=8，LR=1e-4，约 89 秒；未搜索超参数，未用 test 选模型。
- 训练 loss（每 5 步日志）0.7856 → 0.6316 → 0.4592 → 0.3877 → 0.3189。
- 同一开发验证集、同一 tokenizer/prompt 前缀的 token 加权 completion NLL：基座 0.8768，LoRA 0.5830，563 个目标 token。
- NLL 是 teacher-forced token loss，不是答案准确率。验证集只有 3 道题，不应外推泛化收益；也没有 7B LoRA 或 DPO 训练效果结论。
- TRL 的 eval_loss=0.4844 是其自身聚合口径，独立验证脚本明确按目标 token 加权，不能混成同一个指标。
- 实际 TRL 数据处理与 collator 审计：78 条，45,073 个 prompt/observation token 全部 mask，2,629 个 completion token 参与监督。
- 检查实际最长序列为 1,473 tokens，全部低于 4,096，没有因训练长度截断目标。
- 新版 Transformers 默认 chat template 返回 BatchEncoding；项目显式指定 return_dict=False 做长度/前缀检查。本次训练后复核了全部 78 条长度，确认未受影响。
- 本地 Agent 适配器已跑过一次独立数学工具演示，4 次调用预算耗尽而失败。原始失败演示保留，不用它证明能力提升，不重复采样到正确。

权重保留在 outputs/lora-qwen05b-20261025/adapter。Git 中保存训练数据、配置、权重校验值和验证结果；不把 1GB 基座权重放入仓库。最终归档报告在 reports 下给出。

## 复现注意

使用单独的 CUDA 环境，实际验证的软件版本会记录在训练归档中。当前宿主 Python 的 numpy 与部分旧编译库不兼容，因此训练放在 .cache/training-venv，使用 numpy 1.26.4 + pyarrow 23.0.1；评测环境未被修改。

```powershell
python -m post_training.train_sft --data <归档的sft.jsonl> --output outputs/my-pilot --model Qwen/Qwen2.5-0.5B-Instruct --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 --max-steps 25 --max-length 4096
python -m post_training.audit_mask --data <归档的sft.jsonl> --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --output outputs/mask-audit.json
python -m post_training.evaluate_sft --data <归档的sft.jsonl> --adapter outputs/my-pilot/adapter --model Qwen/Qwen2.5-0.5B-Instruct --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 --output outputs/validation-comparison.json
```

本地模型可通过 python -m agent.run --backend local --model ... --adapter ... 接入同一个 ReAct 循环。任意模型接入成功不意味着特定任务一定答对。

