# 真实复测报告

- Source revision: `d95783ab0befb67c875ab6c9c0d817eb4dc9ae07`
- Protocol: `one-trajectory-no-answer-selection`; split: `dev`
- Scorer: `strict-v2`; inference source fingerprint: `legacy: Git revision and clean-start flag`
- Progress: 4/4 episodes; complete: True
- Seeds requested: [29]; model: `Qwen/Qwen2.5-7B-Instruct`
- QA retrieval: `live`; context hash: `None`

每个题目、种子、模式只运行一条轨迹，所有失败计入分母。success 仅表示产出了协议合法的最终答案。

| Task | Mode | Correct / N | Accuracy | Answer F1 | Avg calls | Avg seconds |
|---|---|---:|---:|---:|---:|---:|
| QA | off | 1/2 | 50.00% | 0.750 | 3.50 | 17.34 |
| QA | on | 0/2 | 0.00% | 0.250 | 5.50 | 11.95 |

## 成对差异与误差分析

- QA: ON − OFF = -50.00 pp; question-cluster bootstrap 95% [-100.00, 0.00] pp; improved=0, regressed=1.
- QA/off: parse errors=0, tool errors=0, terminal failures={}, tool failure types={}.
- QA/on: parse errors=1, tool errors=7, terminal failures={'max_calls_reached': 1}, tool failure types={'retrieval_http_429': 7}.

## 解释边界

- 重复种子不能当作独立的新题；主差异区间按题目聚类重采样。每种子 Wilson 区间在 summary.json 中。
- API seed 为请求参数，不能保证服务商确定性；未做择优、重试到正确或基于答案的提前停止。
- 数据来自原项目，曾用于历史调试；test 是本次冻结的操作划分，不宣称完全未见过。
- QA 使用 manifest 中声明的检索后端；snapshot 为问题对应的固定候选段落上的 BM25，live 为实时 Wikipedia。只评答案 EM/F1，不评 supporting facts / joint score，不能对标官方完整榜单。
- 旧日志存在人工挑选，且协议、判分和实现均有变化，本报告不能用于推断相对旧版本的因果增益。
- API/工具故障保留并计为失败；费用以供应商账单为准，日志只记录 token 用量。
- 区间跨零时，不应声称反思有显著增益；即使不跨零，也仅适用于这里的数据和配置。
