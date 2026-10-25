# 真实复测报告

- Source revision: `42c0487e9430b98bffd06ebff8f0a3f8a2185b81`
- Protocol: `one-trajectory-no-answer-selection`; split: `test`
- Progress: 28/1290 episodes; complete: False
- Seeds requested: [11, 29, 47]; model: `Qwen/Qwen2.5-7B-Instruct`

每个题目、种子、模式只运行一条轨迹，所有失败计入分母。success 仅表示产出了协议合法的最终答案。

| Task | Mode | Correct / N | Accuracy | Answer F1 | Avg calls | Avg seconds |
|---|---|---:|---:|---:|---:|---:|
| MATH | off | 0/4 | 0.00% | — | 7.00 | 67.59 |
| MATH | on | 2/4 | 50.00% | — | 2.75 | 12.32 |
| QA | off | 6/13 | 46.15% | 0.558 | 3.46 | 104.88 |
| QA | on | 1/7 | 14.29% | 0.190 | 4.14 | 133.29 |

## 成对差异与误差分析

- MATH: ON − OFF = 0.00 pp; question-cluster bootstrap 95% [0.00, 0.00] pp; improved=0, regressed=0.
- QA: ON − OFF = 0.00 pp; question-cluster bootstrap 95% [0.00, 0.00] pp; improved=0, regressed=0.
- MATH/off: parse errors=18, tool errors=4, terminal failures={'max_calls_reached': 1}, tool failure types={'unknown_tool': 1, 'execution_error': 3}.
- MATH/on: parse errors=3, tool errors=0, terminal failures={}, tool failure types={}.
- QA/off: parse errors=1, tool errors=3, terminal failures={'max_calls_reached': 1}, tool failure types={'retrieval_http_429': 3}.
- QA/on: parse errors=3, tool errors=4, terminal failures={'api_error': 2}, tool failure types={'retrieval_http_429': 4}.

## 解释边界

- 重复种子不能当作独立的新题；主差异区间按题目聚类重采样。每种子 Wilson 区间在 summary.json 中。
- API seed 为请求参数，不能保证服务商确定性；未做择优、重试到正确或基于答案的提前停止。
- 数据来自原项目，曾用于历史调试；test 是本次冻结的操作划分，不宣称完全未见过。
- QA 是实时 Wikipedia 检索任务，仅使用 HotpotQA 风格答案 EM/F1，没有 supporting facts / joint score，不能对标官方榜单。
- 旧日志存在人工挑选，且协议、判分和实现均有变化，本报告不能用于推断相对旧版本的因果增益。
- API/工具故障保留并计为失败；费用以供应商账单为准，日志只记录 token 用量。
- 区间跨零时，不应声称反思有显著增益；即使不跨零，也仅适用于这里的数据和配置。
