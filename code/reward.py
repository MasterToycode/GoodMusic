import re
import numpy as np
from typing import List, Dict, Union, Optional
from sentence_transformers import SentenceTransformer, util
from multiprocessing import Pool, cpu_count

# 全局初始化 SentenceTransformer 模型，并移动到 GPU
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to("cuda")


def compute_rewards(
        completions: List[str],
        min_len: Union[int, List[int]] = 100,
        max_len: Union[int, List[int]] = 300,
        weights: Union[tuple, List[tuple]] = (0.25, 0.25, 0.25, 0.25),
        return_components: bool = False,
        **kwargs
) -> Union[List[float], Dict[str, List[float]]]:
    """并行优化的奖励计算函数"""
    keywords = kwargs["keywords"]
    n_samples = len(completions)

    min_len = _to_list(min_len, n_samples)
    max_len = _to_list(max_len, n_samples)
    weights = _to_list(weights, n_samples)

    # 并行计算各子奖励
    with Pool(cpu_count()) as pool:
        length_rewards = pool.starmap(_length_reward, zip(completions, min_len, max_len))
        format_rewards = pool.map(_format_reward, completions)
        keyword_rewards = _batch_keyword_reward(completions, keywords)  # 这个用 GPU 计算
        language_rewards = pool.map(_language_reward, completions)

    # 加权求和总奖励
    total_rewards = [
        w[0] * lr + w[1] * fr + w[2] * kr + w[3] * lang_r
        for w, lr, fr, kr, lang_r in zip(weights, length_rewards, format_rewards, keyword_rewards, language_rewards)
    ]

    if return_components:
        return {
            "rewards": total_rewards,
            "length_rewards": length_rewards,
            "format_rewards": format_rewards,
            "keyword_rewards": keyword_rewards,
            "language_rewards": language_rewards,
        }
    return total_rewards


# -------------- 并行子函数 --------------
def _to_list(val: Union[any, List[any]], n: int) -> List[any]:
    """转换为样本级列表"""
    return val if isinstance(val, list) else [val] * n


def _length_reward(text: str, min_len: int, max_len: int) -> float:
    """单样本长度奖励"""
    original = text.split("</think>:", 1)[1].strip() if "</think>:" in text else text.strip()
    length = len(original)

    if length < min_len:
        return length / min_len + 1  # 1~2线性增长
    elif length > max_len:
        return max_len / length + 1  # 2~1线性衰减
    return 2.0


def _format_reward(text: str) -> float:
    """单样本格式奖励"""
    if "<think>" not in text or "</think>:" not in text:
        return -2.0
    think_content = text.split("<think>")[1].split("</think>")[0].strip()
    return 2.0 if think_content else -2.0


def _batch_keyword_reward(texts: List[str], keywords_list: List[List[str]]) -> List[float]:
    """批量关键词匹配（优化：使用 GPU 并行计算）"""
    originals = [text.split("</think>:", 1)[1].strip() if "</think>:" in text else text.strip() for text in texts]
    valid_indices = [i for i, orig in enumerate(originals) if orig and keywords_list[i]]

    if not valid_indices:
        return [0.8 if not kw else -2.0 for kw in keywords_list]  # 无关键词时默认0.8

    valid_originals = [originals[i] for i in valid_indices]
    valid_keywords = [keywords_list[i] for i in valid_indices]

    # 让计算在 GPU 上执行
    original_embs = embedder.encode(valid_originals, convert_to_tensor=True)
    keyword_embs = [embedder.encode(kw, convert_to_tensor=True) for kw in valid_keywords]

    similarities = [
        util.pytorch_cos_sim(orig_emb, kw_emb).mean().item()
        for orig_emb, kw_emb in zip(original_embs, keyword_embs)
    ]

    # 分配奖励
    rewards = []
    sim_idx = 0
    for i, kw in enumerate(keywords_list):
        if i in valid_indices:
            sim = similarities[sim_idx]
            rewards.append(2.0 if sim >= 0.6 else (1.2 if sim >= 0.4 else 0.8))
            sim_idx += 1
        else:
            rewards.append(0.8 if not kw else -2.0)
    return rewards


def _language_reward(text: str) -> float:
    """单样本语言奖励"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    ratio = chinese_chars / max(1, len(text))

    if ratio >= 0.9:
        return 2.0
    elif ratio >= 0.7:
        return 1.4
    return 0.7


# ------------ 运行示例 ------------
if __name__ == "__main__":
    samples = [
        "科技<think>技术创新是关键</think>:人工智能在医疗领域的应用正在改变诊断方式。",
        "无效样本<think></think>:无意义内容",
        "经济<think>宏观经济分析</think>:全球供应链重构对发展中国家影响深远。"
    ]
    keywords = [
        ["科技", "人工智能"],
        [],  # 空关键词
        ["经济", "供应链"]
    ]

    # 并行计算
    rewards = compute_rewards(
        completions=samples,
        keywords=keywords,
        min_len=[50, 10, 80],
        return_components=True
    )

    print("总奖励:", rewards["rewards"])
    print("长度奖励:", rewards["length_rewards"])
    print("格式奖励:", rewards["format_rewards"])
    print("关键词奖励:", rewards["keyword_rewards"])
    print("语言奖励:", rewards["language_rewards"])
