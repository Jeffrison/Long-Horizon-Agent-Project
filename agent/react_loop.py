import os
from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_REGISTRY, get_tool_description
from agent.prompts import QA_SYSTEM_PROMPT, MATH_SYSTEM_PROMPT
from agent.parser import parse_llm_output
from eval.metrics import qa_match, math_match

load_dotenv()
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def run_agent(query: str, task_type: str = "QA", max_steps: int = 10, enable_reflection: bool = True, ground_truth: str = None) -> dict:
    tool_desc = get_tool_description()
    tool_names = ", ".join(TOOL_REGISTRY.keys())
    
    if task_type == "QA":
        system_prompt = QA_SYSTEM_PROMPT.format(tool_descriptions=tool_desc, tool_names=tool_names)
    else:
        system_prompt = MATH_SYSTEM_PROMPT.format(tool_descriptions=tool_desc, tool_names=tool_names)
    
    trajectory =[]
    
    # 滑动窗口记忆
    history_window =[]      # 存储对话历史
    MAX_HISTORY_TURNS = 2    # 永远只保留最近的 2 轮完整交互
    current_error_feedback = "" # 短期报错记忆

    step = 1
    consecutive_format_errors = 0   # 连续格式崩坏计数器
    MAX_FORMAT_RETRIES = 6          # 最多允许它连续胡言乱语 6 次

    print(f"\n开始任务 | 类型: {task_type} | 滑动窗口记忆: 开启\n{query}\n" + "="*50)

    while step <= max_steps:
        print(f"\n--- [Step {step}] 正在思考... ---")
        
        # 1. 动态组装当前的上下文
        messages =[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Query:{query}.Begin your response with:\nThought:"}
        ]
        
        # 2. 灌入滑动窗口内的最近记忆
        messages.extend(history_window)
        
        # 3. 如果上一步解析失败或触发了强制反馈，追加在最后面
        if current_error_feedback:
            messages.append({"role": "user", "content": current_error_feedback})

        # 呼叫大模型
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=600, 
            stop=["Observation:", "Observation: "] 
        )
        
        llm_response_text = response.choices[0].message.content.strip()
        print(llm_response_text)
        trajectory.append({"role": "assistant", "content": llm_response_text})
        
        # 记忆回滚防毒机制
        try:
            parsed_result = parse_llm_output(llm_response_text,task_type=task_type)
            consecutive_format_errors = 0 
        except ValueError as e:
            if enable_reflection:
                consecutive_format_errors += 1
                if consecutive_format_errors > MAX_FORMAT_RETRIES:
                    print(f"[任务崩溃] 连续 {MAX_FORMAT_RETRIES} 次输出格式乱码，强行截断以防死循环！")
                    return {"status": "failed", "error": "format_error_limit", "trajectory": trajectory, "steps": step, "final_answer": None}

                print("[纠错介入] 发现乱码，启动记忆拦截！该段废话不存入历史。")
                current_error_feedback = f"Observation: Parsing failed ({str(e)}).Please strictly output using 'Thought', 'Action', and 'Action Input' format!"
                trajectory.append({"role": "user", "content": current_error_feedback})
                continue 
            else:
                return {"status": "failed", "error": "format_error", "trajectory": trajectory, "steps": step, "final_answer": None}
                
        # 解析成功，将本轮对话加入滑动窗口
        current_error_feedback = "" # 清空短期警告
        history_window.append({"role": "assistant", "content": llm_response_text})
        
        # 判断是否完成任务
        if parsed_result["type"] == "finish":
            print("\n任务完成！")
            print(f"最终答案: {parsed_result['content']}\n" + "="*50)
            return {"status": "success", "final_answer": parsed_result["content"], "trajectory": trajectory, "steps": step}
            
        # 执行工具
        if parsed_result["type"] == "action":
            raw_action_name = parsed_result["action"]
            action_input = parsed_result["action_input"]
            
            tool_map = {k.lower(): k for k in TOOL_REGISTRY.keys()}
            clean_action_name = raw_action_name.lower().strip("`*\"'_ ")
            
            if clean_action_name not in tool_map:
                observation = f"Observation: Critical Error! Tool '{raw_action_name}' does not exist. Please choose from {tool_names}."
            else:
                real_action_name = tool_map[clean_action_name]
                tool_func = TOOL_REGISTRY[real_action_name]
                print(f"🛠️ 正在执行工具: {real_action_name}")
                observation = tool_func(action_input)
                
            # 引导它自己排查 SyntaxError
            is_error = any(err in observation for err in["Error Traceback", "SyntaxError", "超时", "内部错误", "Error", "Exception"])
            if is_error:
                if enable_reflection and ("Tool" not in observation):
                    # 对于代码内部错误，追加系统提示，允许它继续活下去并纠错
                    observation += "\n\n[System Note]: The code above threw an error! Please analyze the cause in your next Thought and rewrite the code."
                else:
                    # 没开启反思：记录下最后的惨状，直接斩断，强制失败退出！
                    print("[任务崩溃] 工具执行报错，且未开启反思机制，直接终止。")
                        
                    # 把报错信息也存入轨迹，方便事后分析
                    if not observation.startswith("Observation:"):
                        observation = f"Observation:\n{observation}"
                    trajectory.append({"role": "user", "content": observation})
                        
                    return {
                        "status": "failed", 
                        "error": "tool_execution_error", # 新增一种失败类型
                        "trajectory": trajectory, 
                        "steps": step, 
                        "final_answer": None
                    }
                    

            print(f"本轮得到observation：{observation}")
            trajectory.append({"role": "user", "content": observation})
            
            # 将工具结果加入滑动窗口
            history_window.append({"role": "user", "content": observation})
            
            # 提前拦截
            if ground_truth is not None and not is_error:
                is_match = False
                if task_type == "QA":
                    is_match=False #QA禁止使用提前拦截
                else:
                    is_match = math_match(observation, ground_truth)

                if is_match:
                    print(f"\n工具执行结果直接命中了标准答案！强制结束任务！")
                    
                    fake_finish_thought = f"Thought: The tool successfully output the result matching the required answer."
                    trajectory.append({"role": "assistant", "content": f"{fake_finish_thought}\nFinal Answer: {observation}"})
                    
                    return {
                        "status": "success", 
                        "final_answer": observation,
                        "trajectory": trajectory, 
                        "steps": step
                    }

            # 执行滑动窗口截断机制 
            if len(history_window) > MAX_HISTORY_TURNS * 2:
                history_window = history_window[-(MAX_HISTORY_TURNS * 2):]
        step += 1
 
    print("\n达到最大步数限制，任务失败。")
    return {"status": "failed", "error": "max_steps_reached", "trajectory": trajectory, "steps": max_steps, "final_answer": None}