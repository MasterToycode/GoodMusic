import os
import openai
from openai import APIError
from typing import Dict, List, Union

# 自定义异常
class InsufficientBalanceError(Exception):
    pass

class EvaluationError(Exception):
    pass

# 系统提示词 - 更详细的评分标准
SYS_PROMPT = """你是一个专业的文本质量评估专家。请根据以下标准对文本进行评分(满分10分):
1. 创意性(权重25%): 内容的原创性和新颖性
2. 文采(权重25%): 语言表达的优美程度和修辞手法
3. 格式(权重25%): 结构清晰度、可读性和符合要求的格式
4. 长度(权重25%): 内容长度是否适中(50-300字为佳)
5. 总分(根据四个维度进行加权计算)

评分要求:
- 使用表格形式输出,得到得分表格.
- 每项评分保留1位小数
- 最后简要对目标文本的评价，而不是让你自己再写一个，切记
"""

def evaluate_text_quality(
    text: str,
    api_key: str = None,
    model: str = "deepseek-chat",
    temperature: float = 0.3,
    max_tokens: int = 300
) -> Dict[str, Union[float, str]]:
    # 获取API密钥
    api_key = api_key or os.getenv("you_api_key")
    if not api_key:
        raise ValueError("DeepSeek API密钥未提供")
    
    # 创建客户端
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )
    
    try:
        # 调用API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        
        # 解析结果
        output = response.choices[0].message.content.strip()
        
        # 从API响应中提取评分
        return parse_evaluation_result(output)
        
    except APIError as e:
        if e.status_code == 402:  # 假设402为余额不足状态码
            raise InsufficientBalanceError("API余额不足,请充值") from e
        else:
            raise EvaluationError(f"API错误[{e.status_code}]: {e.message}") from e
    except Exception as e:
        raise EvaluationError(f"评估失败: {str(e)}") from e

def parse_evaluation_result(output: str) -> Dict[str, Union[float, str]]:
    """
    改进后的评估结果解析函数，能更好处理中文评分表格
    """
    result = {
        "scores": {
            "creativity": 0.0,
            "language": 0.0,
            "format": 0.0,
            "length": 0.0,
            "total": 0.0
        },
        "evaluation": output  # 默认保留全部输出
    }
    
    # 改进的表格解析逻辑
    lines = [line.strip() for line in output.split('\n') if line.strip()]
    
    for line in lines:
        # 处理创意性评分
        if "创意性" in line:
            result["scores"]["creativity"] = extract_score_from_line(line)
        # 处理文采评分
        elif any(key in line for key in ["文采", "语言表达"]):
            result["scores"]["language"] = extract_score_from_line(line)
        # 处理格式评分
        elif "格式" in line:
            result["scores"]["format"] = extract_score_from_line(line)
        # 处理长度评分
        elif "长度" in line:
            result["scores"]["length"] = extract_score_from_line(line)
        # 处理总分
        elif any(key in line for key in ["总分", "总计", "平均"]):
            result["scores"]["total"] = extract_score_from_line(line)
    
    # 提取评价部分（从"评价："之后的内容）
    evaluation_lines = []
    found_evaluation = False
    for line in lines:
        if any(prefix in line for prefix in ["评价：", "评语：", "总结："]):
            found_evaluation = True
            line = line.split("：", 1)[-1].strip()
        if found_evaluation and line:
            evaluation_lines.append(line)
    
    if evaluation_lines:
        result["evaluation"] = "\n".join(evaluation_lines)
    
    return result

def extract_score_from_line(line: str) -> float:
    """
    改进的分数提取函数，能处理多种表格格式
    """
    try:
        # 处理 | 创意性 | 8.5 | 这种格式
        if "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            for part in parts:
                if part.replace('.', '').isdigit():
                    return float(part)
        
        # 处理 "创意性: 8.5" 这种格式
        if ":" in line or "：" in line:
            parts = line.split(":", 1) if ":" in line else line.split("：", 1)
            num_part = parts[-1].strip()
            for s in num_part.split():
                s = s.replace('/', '').replace('分', '')
                if s.replace('.', '').isdigit():
                    return float(s)
        
        # 直接搜索数字
        for word in line.split():
            word = word.replace('分', '').replace('/', '')
            if word.replace('.', '').isdigit():
                return float(word)
                
    except (ValueError, IndexError):
        pass
    
    return 0.0

    
def print_evaluation_result(
    evaluation: Dict[str, Union[float, str]],
    show_details: bool = True,
    score_only: bool = False
) -> None:
    """
    打印评估结果
    
    参数:
        evaluation: evaluate_text_quality返回的评估结果字典
        show_details: 是否显示详细评价
        score_only: 是否仅显示分数(优先级高于show_details)
    """
    if not evaluation:
        print("无有效评估结果")
        return
    
    scores = evaluation.get("scores", {})
    evaluation_text = evaluation.get("evaluation", "")
    
    # 打印分数摘要
    print("\n=== 文本质量评估 ===")
    print(f"[创意性] {scores.get('creativity', 0.0):.1f}/10")
    print(f"[文采]    {scores.get('language', 0.0):.1f}/10")
    print(f"[格式]    {scores.get('format', 0.0):.1f}/10")
    print(f"[长度]    {scores.get('length', 0.0):.1f}/10")
    print("-" * 25)
    print(f"[总分]    {scores.get('total', 0.0):.1f}/10")
    
    # 根据参数决定是否显示详细评价
    if not score_only and show_details and evaluation_text:
        print("\n=== 详细评价 ===")
        print(evaluation_text)