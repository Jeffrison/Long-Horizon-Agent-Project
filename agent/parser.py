import re

def check_gibberish(text: str):
    """
    乱码拦截，一旦发现乱码抛出 ValueError。
    """
    if not text:
        return
        
    # 检查连续相同的单词
    words = text.split()
    if len(words) > 5:
        max_consec, curr_consec = 1, 1
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                curr_consec += 1
                max_consec = max(max_consec, curr_consec)
            else:
                curr_consec = 1
        if max_consec >= 20: # 如果同一个单词连续出现 20 次以上
            raise ValueError("Repetitive words detected. Your logic has degenerated. Please rewrite clearly.")

    # 检查大块连续重复的字符串
    if re.search(r'(\S.{0,20}?)\1{19,}', text, flags=re.DOTALL):
        raise ValueError("Repetitive pattern detected!Please rethink!")
def parse_llm_output(text: str,task_type: str = "QA") -> dict:
    # 先做清理
    clean_text = text.strip()

    # 改进的 Final Answer 提取，不区分大小写
    # 使用正则 re.IGNORECASE 查找 "Final Answer:"
    final_match = re.search(r"Final\s+Answer:\s*(.*)", clean_text, re.IGNORECASE | re.DOTALL)
    
    if final_match:
        # 提取匹配到的内容并取第一行
        answer_content = final_match.group(1).strip().split('\n')[0].strip()
        # 去掉末尾可能存在的反括号或多余标点
        answer_content = answer_content.rstrip(').! ')

        check_gibberish(answer_content)

        if task_type == "MATH":
                try:
                    # 尝试将其转为浮点数（先去掉可能的千分位逗号，如 1,000）
                    float(answer_content.replace(',', ''))
                except ValueError:
                    # 如果转换失败（比如它包含了英文字母或废话），直接抛出异常拦截！
                    raise ValueError(
                        f"You outputted: '{answer_content}'. "
                        f"Please rethink,your Final Answer MUST be a pure decimal floating-point number."
                    )

        return {"type": "finish", "content": answer_content}

    # Action 提取保持之前的大小写忽略正则
    action_match = re.search(r"Action:\s*(.*?)\n", clean_text + "\n", re.IGNORECASE)
    # re.DOTALL 使得 .* 可以匹配换行符，因为代码可能是多行的
    input_match = re.search(r"Action Input:\s*(.*)", text, re.IGNORECASE | re.DOTALL)

    if action_match and input_match:
        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip()
        
        # 抹除所有 Markdown 痕迹
        # 匹配并抹除所有带语言标识的开头，例如 ```python, ```bash, ``` 甚至是代码中间的这些符号
        action_input = re.sub(r"```[a-zA-Z]*", "", action_input)
        # 抹除所有剩余的反引号 ``` 
        action_input = action_input.replace("```", "")
        # 去除首尾的多余空格、换行符和首尾的普通引号
        action_input = action_input.strip("\"' \n")

        # 拦截空输入
        if not action_input:
             raise ValueError("Your Action Input is empty! You must provide something.")
        check_gibberish(action_input)

        return {
            "type": "action", 
            "action": action, 
            "action_input": action_input
        }

    # 如果既没有 Final Answer也没有 Action，我们直接抛出 ValueError，在下个阶段的主循环中，我们会捕获这个错误并喂回给模型
    raise ValueError(f"Parsing failed! Could not find valid 'Action' and 'Action Input'. Please check your output format.\nYour raw output was: {text}")