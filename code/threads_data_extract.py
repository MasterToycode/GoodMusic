import os
import openai
import threading
from concurrent.futures import ThreadPoolExecutor
from openai import APIError
from typing import List, Tuple

class FormatValidator:
    """数据格式验证器"""
    @staticmethod
    def validate_line(keywords: List[str], original: str) -> str:
        """
        格式：关键词1，关键词2，关键词3：原文
        """
        # 清洗关键词中的非法符号
        cleaned_keywords = [
            kw.strip().replace('：', '').replace('\n', '')[:10]  # 限制关键词长度
            for kw in keywords if kw.strip()
        ][:3]  # 最多取前3个关键词
        
        # 处理空关键词情况
        if not cleaned_keywords:
            keywords_str = "无关键词"
        else:
            keywords_str = "，".join(cleaned_keywords)
        
        # 移除原文中的换行符
        cleaned_original = original.strip().replace('\n', ' ')
        return f"{keywords_str}：{cleaned_original}"

class ThreadSafeWriter:
    """增强型线程安全写入器"""
    def __init__(self, output_path: str):
        self.file = open(output_path, 'a+', encoding='utf-8')
        self.lock = threading.Lock()
        self.counter = 0  # 写入计数器
    
    def write_line(self, content: str):
        with self.lock:
            self.file.write(content + '\n')
            self.file.flush()
            self.counter += 1
    
    def get_progress(self):
        with self.lock:
            return self.counter
    
    def close(self):
        self.file.close()

class DeepSeekBatchProcessor:
    def __init__(self, max_workers: int = 100):
        self.client = openai.OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", "sk-4da7e956235447e3b7bec1b51f5a3db7"),
            base_url="https://api.deepseek.com"
        )
        self.max_workers = max_workers
        self.error_flag = threading.Event()
        self.rate_limiter = threading.Semaphore(20)  # API速率限制

    def process_batch(self, batch: List[Tuple[int, str]], writer: ThreadSafeWriter):
        """批量处理并保持顺序"""
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for line_num, original in batch:
                if self.error_flag.is_set():
                    break
                futures.append(
                    executor.submit(
                        self._process_single_line,
                        line_num,
                        original,
                        writer
                    )
                )
            
            # 等待当前批次完成
            for future in futures:
                future.result()

    def _process_single_line(self, line_num: int, original: str, writer: ThreadSafeWriter):
        if self.error_flag.is_set():
            return

        retries = 0
        while retries < 3 and not self.error_flag.is_set():
            try:
                with self.rate_limiter:
                    response = self.client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[
                            {"role": "system", "content": self._get_prompt()},
                            {"role": "user", "content": original}
                        ],
                        temperature=0.1,
                        max_tokens=30
                    )
                
                # 解析响应
                keywords = self._parse_response(response)
                formatted_line = FormatValidator.validate_line(keywords, original)
                writer.write_line(formatted_line)
                
                # 更新进度
                progress = writer.get_progress()
                print(f"\r已处理 {progress} 条", end='')
                
                break  # 成功时退出重试循环

            except APIError as e:
                if e.status_code == 402:  # 余额不足
                    print(f"\n行 {line_num} 处理失败：API余额不足")
                    self.error_flag.set()
                    return
                elif e.status_code == 429:  # 速率限制
                    print(f"\n行 {line_num} 速率受限，重试中...")
                    retries += 1
                    if retries >= 3:
                        print(f"行 {line_num} 重试次数耗尽")
                else:
                    print(f"\n行 {line_num} API错误[{e.status_code}]：{e.message}")
                    return  # 其他API错误不重试

            except Exception as e:
                print(f"\n行 {line_num} 处理异常：{str(e)}")
                retries += 1
                if retries >= 3:
                    print(f"行 {line_num} 重试次数耗尽")
        
        # 重试失败处理
        if retries >= 3 and not self.error_flag.is_set():
            writer.write_line(f"处理失败：{original}")  # 记录失败行

    @staticmethod
    def _get_prompt() -> str:
        return 

    @staticmethod
    def _parse_response(response) -> List[str]:
        content = response.choices[0].message.content.strip()
        return [kw.strip("。、") for kw in content.replace('，', ',').split(',') if kw]

def process_large_file(
    input_path: str,
    output_path: str,
    batch_size: int = 500,
    max_workers: int = 100
):
    """大文件处理入口"""
    # 初始化组件
    processor = DeepSeekBatchProcessor(max_workers)
    writer = ThreadSafeWriter(output_path)
    
    try:
        # 读取并批处理数据
        with open(input_path, 'r', encoding='utf-8') as f:
            # 生成带行号的批次 [(行号, 内容), ...]
            batches = []
            current_batch = []
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    current_batch.append( (line_num, line.strip()) )
                    if len(current_batch) >= batch_size:
                        batches.append(current_batch)
                        current_batch = []
            if current_batch:
                batches.append(current_batch)

        # 按批次处理（保持批次顺序）
        total = sum(len(b) for b in batches)
        print(f"总数据量：{total}条")
        
        for batch in batches:
            if processor.error_flag.is_set():
                break
            processor.process_batch(batch, writer)
        
        print("\n处理完成！")
    
    finally:
        writer.close()

if __name__ == '__main__':
    # 文件路径配置
    input_file = "data\DSdata.txt"
    output_file = "data\CoTdata.txt"
    
    # 启动处理流程
    process_large_file(
        input_path=input_file,
        output_path=output_file,
        batch_size=500,
        max_workers=100
    )