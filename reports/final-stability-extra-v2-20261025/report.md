# 真实复测报告

- Source revision: `44694c5fd9233a6b800b239d13448f4dcff4ba02`
- Protocol: `one-trajectory-no-answer-selection`; split: `test`
- Progress: 80/80 episodes; complete: True
- Seeds requested: [29, 47]; model: `Qwen/Qwen2.5-7B-Instruct`
- QA retrieval: `snapshot`; context hash: `e18c10fcffd0a7225e88dac89268773360594ca9cd7a9e41afbdb373629136ac`

每个题目、种子、模式只运行一条轨迹，所有失败计入分母。success 仅表示产出了协议合法的最终答案。

| Task | Mode | Correct / N | Accuracy | Answer F1 | Avg calls | Avg seconds |
|---|---|---:|---:|---:|---:|---:|
| MATH | off | 3/20 | 15.00% | — | 6.10 | 78.11 |
| MATH | on | 4/20 | 20.00% | — | 4.80 | 75.15 |
| QA | off | 6/20 | 30.00% | 0.503 | 3.55 | 21.47 |
| QA | on | 6/20 | 30.00% | 0.500 | 3.30 | 16.84 |

## 成对差异与误差分析

- MATH: ON − OFF = 5.00 pp; question-cluster bootstrap 95% [0.00, 15.00] pp; improved=1, regressed=0.
- QA: ON − OFF = 0.00 pp; question-cluster bootstrap 95% [-15.00, 15.00] pp; improved=1, regressed=1.
- MATH/off: parse errors=87, tool errors=13, terminal failures={'max_calls_reached': 9}, tool failure types={'invalid_code': 5, 'unknown_tool': 2, 'execution_error': 6}.
- MATH/on: parse errors=26, tool errors=20, terminal failures={'max_calls_reached': 3, 'api_error': 1}, tool failure types={'timeout': 5, 'execution_error': 14, 'unknown_tool': 1}.
- QA/off: parse errors=17, tool errors=0, terminal failures={'max_calls_reached': 3}, tool failure types={}.
- QA/on: parse errors=17, tool errors=0, terminal failures={'max_calls_reached': 2}, tool failure types={}.

## 解释边界

- 重复种子不能当作独立的新题；主差异区间按题目聚类重采样。每种子 Wilson 区间在 summary.json 中。
- API seed 为请求参数，不能保证服务商确定性；未做择优、重试到正确或基于答案的提前停止。
- 数据来自原项目，曾用于历史调试；test 是本次冻结的操作划分，不宣称完全未见过。
- QA 使用 manifest 中声明的检索后端；snapshot 为问题对应的固定候选段落上的 BM25，live 为实时 Wikipedia。只评答案 EM/F1，不评 supporting facts / joint score，不能对标官方完整榜单。
- 旧日志存在人工挑选，且协议、判分和实现均有变化，本报告不能用于推断相对旧版本的因果增益。
- API/工具故障保留并计为失败；费用以供应商账单为准，日志只记录 token 用量。
- 区间跨零时，不应声称反思有显著增益；即使不跨零，也仅适用于这里的数据和配置。
