# 固定语料检索模式

## 为什么增加这个后端

2026-10-25 的实时检索完整计划在起跑阶段遇到 Wikipedia 429 和请求超时。该轮停止时仅完成 26 / 1290 条轨迹，已经归档为 **incomplete / aborted_infrastructure**，不能当完整准确率估计。停止原因是外部依赖与吞吐，不是为了筛选正确答案。

新增 snapshot 后端用于另一项独立实验：让反思开关两组使用完全相同、预先冻结的候选语料，消除本次网络可用性的直接干扰。它回答的是“固定候选段落检索条件下”的效果，不代表实时网络场景，也不与实时检索结果直接比较高低。

## 数据来源和可见边界

来源：[HotpotQA](https://hotpotqa.github.io/) 的公开 distractor validation 数据，[Hugging Face 快照](https://huggingface.co/datasets/hotpotqa/hotpot_qa)。

- 原项目 200 道 QA 题全部匹配，问题哈希和答案一致性在准备阶段校验。
- 199 题有 10 个候选段落，1 题在该公开快照中只有 2 个；原样保留并披露，不补造 distractors。
- 工具文件仅含 question_key、documents.title、documents.text。
- **不包含 answer、supporting_facts、难度、正确证据标签**；代码校验额外字段时会拒绝加载。
- 原始文件 SHA256 与提取语料 SHA256、字段和许可在 data/contexts/*.metadata.json 中。
- 已冻结的纯文本语料直接随仓库保存，推理无需 PyArrow 或联网取 Wikipedia。

准备阶段可以用答案校验版本是否一致，但传给 Agent 的只有问题与检索工具闭包；答案不进入工具文件或推理消息。

## 检索算法

手写 BM25：k1=1.5、b=0.75，标题重复一次加权，完全匹配标题额外加分 10。每次返回前两个正分候选段落；同分按标题和原索引稳定排序。语料按题目隔离，只索引该题公开给系统的候选段落。

这是一个 **question-scoped candidate retrieval** 环境。它不搜索全量 Wikipedia；也没有预测 supporting facts，所以不能宣称复现完整官方 HotpotQA 系统。

## 使用

```powershell
# 默认后端为 snapshot
python -m eval.run_eval --retrieval snapshot --split test --seeds 11 29 47 --workers 8 --output outputs/my-snapshot-test

# 原来的实时工具仍可独立运行
python -m eval.run_eval --retrieval live --split dev --limit 2 --seeds 11 --workers 1 --output outputs/my-live-check
```

snapshot 的文件哈希纳入 manifest；改变语料后不能恢复旧实验。网络模式每个 HTTP 请求共享节流，429 的 Retry-After 只对后续请求生效，不进行隐藏重试。

## 可选重新准备

运行评测不需要此步骤。要从公开 Parquet 再生成语料，单独安装 requirements-data.txt，用 requests 下载 metadata 中的 source_url，再执行：

```powershell
python -m eval.prepare_contexts <下载的validation.parquet>
```

如果下载内容版本不同，需先核对哈希并创建新版本语料，不能覆盖已声明的实验。

数据遵循 HotpotQA/Wikipedia 的 CC BY-SA 4.0，归属和来源见 metadata 与 DATA_ATTRIBUTION.md。
