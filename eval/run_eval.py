# eval/run_eval.py
import sys
import os
import json
import time
from datetime import datetime

# 确保能导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.data_loader import load_jsonl_dataset
from eval.metrics import qa_match,math_match
from agent.react_loop import run_agent

def run_evaluation(dataset_path: str, task_type: str, max_samples: int = 10, enable_reflection: bool = True):
    """
    task_type: "QA" (HotpotQA) 或 "MATH" (Math Dataset)
    """
    print(f"========== 评测启动 ==========")
    print(f"数据集: {dataset_path}")
    print(f"任务类型: {task_type}")
    print(f"测试数量: {max_samples} 条")
    print(f"纠错机制: {'开启✅' if enable_reflection else '关闭❌'}")
    print(f"==============================\n")

    # 1. 加载数据
    try:
        dataset = load_jsonl_dataset(dataset_path, max_samples=max_samples)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
        
    if not dataset:
        print("数据集为空！请检查数据路径或格式。")
        return

    # 2. 统计指标初始化
    metrics = {
        "total": len(dataset),
        "correct": 0,
        "failed_format": 0,
        "failed_max_steps": 0,
        "total_steps": 0,
        "accuracy": 0.0,
        "avg_steps": 0.0
    }
    
    # 记录详细日志
    detailed_logs =[]
    
    # 3. 循环遍历数据集
    for index, item in enumerate(dataset):
        print(f"\n[{index+1}/{len(dataset)}] 正在测试题目 ID: {item['id']}")
        query = item['question']
        ground_truth = item['answer']
        print(f"Q: {query}")
        print(f"Ground Truth: {ground_truth}")
        
        # 运行智能体
        start_time = time.time()
        agent_result = run_agent(
            query=query, 
            task_type=task_type, 
            max_steps=8, 
            enable_reflection=enable_reflection,
            ground_truth=ground_truth
        )
        time_cost = time.time() - start_time
        
        metrics["total_steps"] += agent_result["steps"]
        is_correct = False
        
        # 4. 判断对错
        if agent_result["status"] == "success":
            pred = agent_result["final_answer"]
            
            if task_type == "QA":
                is_correct = qa_match(pred, ground_truth)
            else:
                is_correct = math_match(pred, ground_truth) 
                
            if is_correct:
                metrics["correct"] += 1
                print(f"✅ [结果判定] 准确！标准答案: {ground_truth}")
            else:
                print(f"❌ [结果判定] 错误。标准答案: {ground_truth}")
        else:
            # 记录失败原因
            error_type = agent_result.get("error", "unknown")
            if error_type == "format_error":
                metrics["failed_format"] += 1
            elif error_type == "max_steps_reached":
                metrics["failed_max_steps"] += 1
            print(f"⚠️ [结果判定] 任务失败 ({error_type})。")

        # 5. 保存单条日志
        log_item = {
            "id": item["id"],
            "question": query,
            "ground_truth": ground_truth,
            "prediction": agent_result.get("final_answer", None),
            "is_correct": is_correct,
            "status": agent_result["status"],
            "steps_taken": agent_result["steps"],
            "time_cost_seconds": round(time_cost, 2),
            "trajectory": agent_result["trajectory"]
        }
        detailed_logs.append(log_item)
        
        # 为了避免触发API 的限流，每跑完一题暂停 2 秒
        time.sleep(2)

    # 6. 计算最终指标
    metrics["accuracy"] = round((metrics["correct"] / metrics["total"]) * 100, 2)
    metrics["avg_steps"] = round(metrics["total_steps"] / metrics["total"], 2)
    
    print("\n" + "#"*40)
    print("📈 最终评测报告")
    print("#"*40)
    print(f"总测试数: {metrics['total']}")
    print(f"答对数量: {metrics['correct']}")
    print(f"准确率 (Accuracy): {metrics['accuracy']}%")
    print(f"平均交互步数: {metrics['avg_steps']} 步")
    print(f"因格式崩坏失败: {metrics['failed_format']} 次")
    print(f"因超时(死循环)失败: {metrics['failed_max_steps']} 次")
    print("#"*40 + "\n")

    # 7. 保存到本地文件
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "Reflect_ON" if enable_reflection else "Reflect_OFF"
    log_filename = f"logs/Eval_{task_type}_{mode_str}_{timestamp}.json"
    
    with open(log_filename, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "logs": detailed_logs
        }, f, ensure_ascii=False, indent=2)
        
    print(f"💾 详细运行日志已保存至: {log_filename}")

if __name__ == "__main__":
    HOTPOT_DATA_PATH = "data/hotpotqa200.jsonl"
    MATH_DATA_PATH = "data/math.jsonl"
    
    # 测试数量
    NUM_SAMPLES = 200
    '''
    print("\n--- 实验 A：关闭纠错机制 ---")
    # 测试数学题，不开启纠错
    run_evaluation(MATH_DATA_PATH, task_type="MATH", max_samples=NUM_SAMPLES, enable_reflection=False)
    '''
    
    print("\n\n--- 实验 B：开启纠错机制 ---")
    # 测试数学题，开启纠错
    
    ''' 
    run_evaluation(MATH_DATA_PATH, task_type="MATH", max_samples=NUM_SAMPLES, enable_reflection=False)
    '''

    run_evaluation(HOTPOT_DATA_PATH, task_type="QA", max_samples=NUM_SAMPLES, enable_reflection=False)
    # QA 任务同理，你可以将上述 MATH 替换为 QA 和 HOTPOT_DATA_PATH 进行测试。
    