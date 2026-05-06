# eval/data_loader.py
import json
import os

def load_jsonl_dataset(file_path: str, max_samples: int = None) -> list:
    """
    file_path: 数据集文件路径
    max_samples: 限制加载的样本数量（用于快速测试调戏）
    返回格式统一的列表:[{'id': '...', 'question': '...', 'answer': '...'}]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到数据集文件: {file_path}。请确保文件已放入正确目录。")

    dataset =[]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line)
                
                question = item.get('question') or item.get('problem')
                answer = item.get('answer')
                item_id = item.get('_id') or item.get('id') or f"sample_{line_num}"
                
                if not question or not answer:
                    print(f"⚠️ 警告: 第 {line_num} 行数据缺少问题或答案字段，已跳过。")
                    continue
                    
                dataset.append({
                    "id": str(item_id),
                    "question": question,
                    "answer": str(answer) # 统一转为字符串，方便后续比对
                })
                
                # 如果设定了最大加载数量，达到后直接停止
                if max_samples and len(dataset) >= max_samples:
                    break
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ 警告: 第 {line_num} 行 JSON 格式错误，已跳过。错误信息: {e}")
                
    return dataset