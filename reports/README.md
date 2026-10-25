# 实验与证据索引

## 最终结果

| 目录 | 作用 |
|---|---|
| [final-seed11-v3-20261025](final-seed11-v3-20261025/report.md) | 全部 215 题、430 条单轨迹成对主实验，严格判分、原始轨迹、错误归因与图表 |
| [final-stability-extra-v3-20261025](final-stability-extra-v3-20261025/report.md) | 预定 20 题、额外 seed=29/47、80 条轨迹 |
| [final-stability-20261025](final-stability-20261025/report.md) | 复用主实验 seed=11 的 40 条，加补充 80 条，分析 20 题的三种子稳定性；没有新采样 |
| [inference-provenance.json](inference-provenance.json) | 验证主/补充实验实际推理模块一致、数据和语料哈希未变 |

主实验 QA OFF/ON 为 42.77%/41.57%，数学为 12.24%/16.33%；差异区间均跨零。正式轨迹共 510 条，11 次 API 超时保留。固定候选语料条件不代表实时网络场景或官方 HotpotQA 完整榜单。

原始 strict-v2 结果分别保存在 final-seed11-v2-20261025 与 final-stability-extra-v2-20261025。strict-v3 对相同预测离线重判，判定变化均为 0；保留父 manifest / 原始记录哈希及 scoring revision。

## 后训练

[lora-qwen05b-20261025](../docs/lora-pilot.md) 保存 78 条 QA SFT 数据、14 条数学导出样本、3 组待审核偏好、训练配置与日志、真实 mask 审计、验证 NLL、权重哈希和失败的本地演示。实际训练只用了 QA 数据，未训练 DPO。权重本体和压缩备份位于 outputs/lora-qwen05b-20261025。

## 前期运行（不是最终性能证据）

| 目录 | 状态与用途 |
|---|---|
| development-smokes/* | 4 轮开发检查，共 20 条轨迹；配置调整与基础设施诊断 |
| live-interrupted-20261025 | 26/1290 完成后因 Wikipedia 429/超时停止；部分任务已开始但未完成，不能估计完整准确率 |
| snapshot-api-stress-20261025 | 1290 计划项全部落盘，但出现大量 HTTP 首步失败；服务压力记录，不当作模型准确率 |
| dev-math-20261025 | 66 条开发数学轨迹，供数据构造，不是 test |
| dev-qa-stress-20261025 | 204 条开发 QA 轨迹，含大量基础设施失败；仅符合过滤条件的正确完整轨迹用于 SFT |
| transport-diagnostics-20261025 | 两次明确不计分的传输诊断，输出未用于任何准确率 |

原项目 logs/ 下四份历史日志有人工挑选偏差，仅保留审计。不同版本和环境不进行逐题择优或最好轮次替换。

## 复核

压缩的 episodes.jsonl.gz 保留每条轨迹、实际 messages、工具输入输出、usage 和失败。manifest 固定题目与配置；verification 检查标签、独立评分和预算；sha256.json 检查单个归档，根目录 artifact-sha256.json 检查全部证据。Git 属性保留 data/logs/reports 的原始字节，避免换行转换破坏校验值。

源代码可通过 Git 历史重建。远端 API seed 不是供应商的确定性保证；旧数据也不是未经调试的新盲测集。

仓库根目录运行 `python -m eval.verify_archives reports` 可核对完整证据清单。
