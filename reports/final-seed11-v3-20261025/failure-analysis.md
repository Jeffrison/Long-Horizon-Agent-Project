# 错误归因

主分类互斥且覆盖全部轨迹；辅助标记可重叠。错误后答对率是条件描述，不能单独作为反思因果增益。

| Task/mode | N | Primary outcomes | Auxiliary flags |
|---|---:|---|---|
| MATH/off | 49 | {'correct': 6, 'max_calls_reached': 24, 'wrong_numeric_final': 17, 'invalid_numeric_final': 2} | {'had_parse_error': 45, 'had_tool_error': 16, 'had_truncation': 26, 'repeated_tool_input': 3, 'error_then_correct': 5} |
| MATH/on | 49 | {'wrong_numeric_final': 36, 'correct': 8, 'invalid_numeric_final': 1, 'max_calls_reached': 4} | {'had_parse_error': 46, 'had_tool_error': 23, 'had_truncation': 16, 'repeated_tool_input': 4, 'error_then_correct': 7} |
| QA/off | 166 | {'qa_partial_match': 34, 'correct': 71, 'max_calls_reached': 15, 'qa_no_token_overlap': 43, 'api_timeout': 3} | {'had_parse_error': 38, 'had_tool_error': 0, 'had_truncation': 0, 'repeated_tool_input': 11, 'error_then_correct': 13} |
| QA/on | 166 | {'qa_no_token_overlap': 46, 'correct': 69, 'qa_partial_match': 32, 'api_timeout': 7, 'max_calls_reached': 12} | {'had_parse_error': 36, 'had_tool_error': 0, 'had_truncation': 0, 'repeated_tool_input': 9, 'error_then_correct': 14} |

成对案例按题目哈希排序选取每类至多 3 条，未按戏剧性筛选。具体文本见 failure-analysis.json。

辅助标记不能相加得到失败总数；API 错误和预算耗尽也可能同时伴随先前的解析/工具错误。