# eval/metrics.py

import re
import string
import math
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    s = ' '.join(s.split())
    return s

def qa_match(prediction: str, ground_truth: str) -> bool:
    """
    字符串匹配
    """
    norm_pred = normalize_text(prediction)
    norm_gt = normalize_text(ground_truth)
    
    if not norm_pred or not norm_gt:
        return False
        
    # 1. 如果标准答案是 "yes"
    if norm_gt == "yes":
        if norm_pred == "yes" or ("yes" in norm_pred and "no " not in norm_pred and "not " not in norm_pred):
            return True
        positive_words =["same", "both", "true", "correct","sure"]
        if any(w in norm_pred for w in positive_words) and "not" not in norm_pred:
            return True
        return False
        
    # 2. 如果标准答案是 "no"
    if norm_gt == "no":
        if norm_pred == "no" or ("no " in norm_pred or "not " in norm_pred):
            return True
        negative_words = ["different", "false", "neither"]
        if any(w in norm_pred for w in negative_words):
            return True
        return False

    # 3. 普通实体题：如果标准答案被完全包含在模型的预测中
    # 例如：GT="YG Entertainment", Pred="it was formed by YG Entertainment"
    if norm_gt in norm_pred:
        return True
        
    return False



def parse_latex_to_float(latex_str: str):
    if not latex_str:
        return None
        
    s = str(latex_str).replace(' ', '').replace('\\\\', '\\')
    
    try:
        return float(s)
    except ValueError:
        pass

    try:
        s = s.replace(r'\pi', 'math.pi')
        s = s.replace(r'\cdot', '*')
        s = s.replace(r'\times', '*')
        
        # =================================================================
        # 【核心重构】：大一统动态坍缩循环 (Unified AST Reduction Loop)
        # 不管怎么嵌套，只要里面有可以脱掉的壳，就一直脱，直到脱不动为止！
        # =================================================================
        prev_s = ""
        while s != prev_s:
            prev_s = s
            # 1. 消除最内层的平方根
            s = re.sub(r'\\sqrt{([^{}]+)}', r'math.sqrt(\1)', s)
            # 2. 消除最内层的乘方
            s = re.sub(r'\^{([^{}]+)}', r'**(\1)', s)
            # 3. 消除最内层的分数
            s = re.sub(r'\\frac{([^{}]+)}{([^{}]+)}', r'((\1)/(\2))', s)
            # 4. 消除阶乘
            s = re.sub(r'([0-9.]+)!', r'math.factorial(\1)', s)
            
        # 兜底：处理没有花括号的简单乘方，例如 x^2
        s = s.replace('^', '**')
        
        # 隐式乘法修正 (如 8math.sqrt -> 8*math.sqrt, 4(2+3) -> 4*(2+3))
        s = re.sub(r'(\d)(math\.sqrt)', r'\1*\2', s)
        s = re.sub(r'(\d)(\()', r'\1*\2', s)
        
        # 将残留的大括号转为小括号（防遗漏）
        s = s.replace('{', '(').replace('}', ')')
        
        allowed_names = {"math": math, "math.pi": math.pi, "math.sqrt": math.sqrt, "math.factorial": math.factorial}
        return float(eval(s, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"Error: {str(e)}"

def math_match(prediction: str, ground_truth: str) -> bool:
    """
    升级版数学判分器：大模型输出浮点数 VS 真实公式
    """
    def extract_pred_number(text: str):
        text = str(text).replace(',', '').strip()
        matches = re.findall(r'-?\d+\.?\d*(?:e[-+]?\d+)?', text, re.IGNORECASE) # 支持科学计数法
        if matches:
            return float(matches[0]) 
        return None

    # 模型预测值通常直接是数字
    pred_num = extract_pred_number(prediction)
    
    # 真实答案可能是 "\frac{1}{576}"，使用我们的解析器转化为 Float
    gt_num = parse_latex_to_float(ground_truth)
    
    # 如果两者都被成功解析为了浮点数，进行容差比对
    if pred_num is not None and gt_num is not None:
        # 容差设为 1e-3 (因为 HMMT 的除法和开方通常在这个精度内)
        # 或者对于科学计数法 (极大数据)，使用相对误差
        if gt_num == 0:
            return abs(pred_num) < 1e-3
        relative_error = abs((pred_num - gt_num) / gt_num)
        absolute_error = abs(pred_num - gt_num)
        return absolute_error < 1e-3 or relative_error < 1e-3

    # 如果无法提取数字（退化为字符串严格匹配）
    return normalize_text(prediction) == normalize_text(ground_truth)