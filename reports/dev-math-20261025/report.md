# 真实复测报告

- Source revision: `42c0487e9430b98bffd06ebff8f0a3f8a2185b81`
- Protocol: `one-trajectory-no-answer-selection`; split: `dev`
- Progress: 66/66 episodes; complete: True
- Seeds requested: [11, 29, 47]; model: `Qwen/Qwen2.5-7B-Instruct`
- QA retrieval: `live`; context hash: `None`

每个题目、种子、模式只运行一条轨迹，所有失败计入分母。success 仅表示产出了协议合法的最终答案。

| Task | Mode | Correct / N | Accuracy | Answer F1 | Avg calls | Avg seconds |
|---|---|---:|---:|---:|---:|---:|
| MATH | off | 4/33 | 12.12% | — | 4.03 | 33.38 |
| MATH | on | 3/33 | 9.09% | — | 2.67 | 17.51 |

## 成对差异与误差分析

- MATH: ON − OFF = -3.03 pp; question-cluster bootstrap 95% [-27.27, 18.18] pp; improved=3, regressed=4.
- MATH/off: parse errors=78, tool errors=8, terminal failures={'api_error': 13, 'max_calls_reached': 4}, tool failure types={'execution_error': 8}.
- MATH/on: parse errors=19, tool errors=13, terminal failures={'api_error': 14}, tool failure types={'unknown_tool': 1, 'execution_error': 5, 'invalid_code': 7}.

## 解释边界

- 重复种子不能当作独立的新题；主差异区间按题目聚类重采样。每种子 Wilson 区间在 summary.json 中。
- API seed 为请求参数，不能保证服务商确定性；未做择优、重试到正确或基于答案的提前停止。
- 数据来自原项目，曾用于历史调试；test 是本次冻结的操作划分，不宣称完全未见过。
- QA 使用 manifest 中声明的检索后端；snapshot 为问题对应的固定候选段落上的 BM25，live 为实时 Wikipedia。只评答案 EM/F1，不评 supporting facts / joint score，不能对标官方完整榜单。
- 旧日志存在人工挑选，且协议、判分和实现均有变化，本报告不能用于推断相对旧版本的因果增益。
- API/工具故障保留并计为失败；费用以供应商账单为准，日志只记录 token 用量。
- 区间跨零时，不应声称反思有显著增益；即使不跨零，也仅适用于这里的数据和配置。
