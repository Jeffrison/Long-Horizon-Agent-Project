# Long-Horizon Agent & Tool-Use from Scratch
本项目是一个不依赖任何重型框架（如 LangChain, AutoGen），从底层纯手写的智能体系统。该 Agent 具备长路径任务处理能力，能够通过 ReAct 循环调用 Wikipedia 搜索和 Python 代码执行工具，并具备强大的自我纠错与工程级鲁棒性。
核心技术亮点
零框架底层实现: 纯 Python 手写 ReAct 主循环，深度理解大模型事件流转机制。
滑动窗口记忆管理: 针对小参数模型（Qwen2.5-7B）设计的记忆策略，仅保留最近交互，防止上下文污染与逻辑断片。
沙箱级代码劫持:
自动拦截 sympy 符号对象并强制转换为浮点数。
基于 ast 静态分析，自动为孤立表达式注入 print()，兼容 Jupyter 编写习惯。
自愈与记忆回滚:
当检测到输出乱码或格式崩溃时，自动拦截“毒性上下文”进入记忆。
通过提示词诱导强行拉回模型的推理分布。
工业级工具健壮性:
Wikipedia: 实现请求节流与 User-Agent 伪装，彻底解决 429 请求频繁报错。
多模态评测对齐: 手写启发式 LaTeX 解析引擎，实现 HMMT/AIME 复杂公式答案与 Python 浮点输出的自动化对齐判分。
# 📂 目录结构
├── agent/
│   ├── prompts.py        # 针对 7B 模型优化的纯英文 System Prompts
│   ├── parser.py         # 带有乱码探测与强类型校验的解析器
│   └── react_loop.py     # 核心主循环（支持反思机制与步数解耦）
├── tools/
│   ├── search_tool.py    # Wikipedia 检索（含 Throttling 机制）
│   └── python_repl.py    # Python 沙箱（含代码劫持与 AST 注入）
├── eval/
│   ├── data_loader.py    # 异构数据集加载器
│   ├── metrics.py        # 自动化 LaTeX 分数判分引擎
│   └── run_eval.py       # 一键评测与 A/B 测试脚本
├── data/                 # 存放评测数据集 (HotpotQA, HMMT, etc.)
├── logs/                 # 存放运行轨迹 (JSON 格式，证明真实可执行)
├── .env.example          # 环境配置模板
├── requirements.txt      # 最小依赖清单
└── README.md             # 本文档

# 🚀 快速开始
1. 环境安装
建议使用 Python 3.9+ 环境：
Bash
git clone https://github.com/Jeffrison/Long-Horizon-Agent-Project.git
cd Long-Horizon-Agent-Project
pip install -r requirements.txt
2. 配置 API Key
复制模板文件并填写你的 API 信息（推荐使用 SiliconFlow 或阿里云百炼）：
Bash
cp .env.example .env
编辑 .env 文件，填入你的 LLM_API_KEY 和 LLM_BASE_URL
3. 运行自动化评测
你可以一键启动针对数学数据集或多跳问答数据集的评测，并观察“纠错机制”开启前后的准确率对比：
Bash
python eval/run_eval.py

# 📊 实验结果分析
在使用 Qwen2.5-7B-Instruct 作为驱动模型时，开启自我纠错带来的增益显著：
数据集	任务类型	准确率 (Reflection OFF)	准确率 (Reflection ON)
HotpotQA	多跳搜索	8.00%	43.00%
HMMT/AIME	代码计算	8.33%	25.00%

# 🛠️ 避坑与反思
Context Pollution: 在开发初期，模型一旦输出乱码，会导致后续对话全部崩坏。通过引入“记忆回滚”机制，系统会拒绝存储格式错误的输出，从而保护了模型思维的纯净度。
LaTeX Alignment: HMMT 数据集的答案 \frac{1}{576} 与 Python 计算结果 0.001736 难以直接匹配。本项目通过手写正则解析器，将符号逻辑转化为数值逻辑，实现了全自动闭环评测。
Rate Limiting: 维基百科 API 的 429 报错曾是批量评测的杀手。通过在 Tool 层引入 1.5s 的步进休眠与规范的 User-Agent，解决了生产环境下的稳定性问题。

# 📜 交付物说明
代码: 严格遵循工程规范，注释详尽。
日志: logs/ 文件夹下包含了完整的模型推理轨迹。
文档: 本 README 提供了完整的复现指引。
# Created by Xuanyi Song for Technical Evaluation Project.