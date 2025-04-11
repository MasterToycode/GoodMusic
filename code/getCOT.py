import os
import openai
import threading
from concurrent.futures import ThreadPoolExecutor
from openai import APIError

API_KEY = os.getenv("DEEPSEEK_API_KEY", "your_api_key")

class ThreadSafeWriter:
    """线程安全写入器"""
    def __init__(self, output_path: str):
        self.file = open(output_path, 'a+', encoding='utf-8')
        self.lock = threading.Lock()
        self.counter = 0

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
            api_key=API_KEY,
            base_url="https://api.deepseek.com/v1"
        )
        self.max_workers = max_workers
        self.error_flag = threading.Event()
        self.rate_limiter = threading.Semaphore(20)

    def process_batch(self, batch, writer: ThreadSafeWriter):
        """批量处理，每个任务单独线程"""
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for line_num, line in batch:
                if self.error_flag.is_set():
                    break
                futures.append(
                    executor.submit(
                        self._process_single_line,
                        line_num,
                        line,
                        writer
                    )
                )
            for future in futures:
                future.result()

    def _process_single_line(self, line_num: int, line: str, writer: ThreadSafeWriter):
        if self.error_flag.is_set():
            return

        # 支持英文冒号(:)和中文全角冒号(：)
        separator = None
        if ':' in line:
            separator = ':'
        elif '：' in line:
            separator = '：'

        if not separator:
            print(f"\n行 {line_num} 格式错误")
            writer.write_line(f"格式错误：{line}")
            return

        keywords_part, original_text = line.split(separator, 1)
        # 这里只提取关键词部分（例如“风，雾，寂寞”）
        keywords = [kw.strip() for kw in keywords_part.split("，") if kw.strip()]
        if not keywords:
            keywords = ["无关键词"]

        # 构造提示：根据关键词生成诗歌
        prompt = "请根据以下关键词写一首诗：" + "，".join(keywords)
        messages = [{"role": "user", "content": prompt}]

        retries = 0
        while retries < 3 and not self.error_flag.is_set():
            try:
                with self.rate_limiter:
                    response = self.client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=messages,
                        temperature=0.1
                    )
                # 提取返回中的思考过程和诗歌原文
                reasoning_content = response.choices[0].message.reasoning_content.replace('\n', '').replace('\r', '')
                poem_original = response.choices[0].message.content.replace('\n', '/').replace('\r', '')
                # 拼接最终结果：关键词<think>思考过程</think>:诗歌原文
                final_line = f"{'，'.join(keywords)}<think>{reasoning_content}</think>:{poem_original}"
                writer.write_line(final_line)
                progress = writer.get_progress()
                print(f"\r已处理 {progress} 条", end='')
                break

            except APIError as e:
                if e.status_code == 402:
                    print(f"\n行 {line_num} 处理失败：API余额不足")
                    self.error_flag.set()
                    return
                elif e.status_code == 429:
                    print(f"\n行 {line_num} 速率受限，重试中...")
                    retries += 1
                    if retries >= 3:
                        print(f"\n行 {line_num} 重试次数耗尽")
                else:
                    print(f"\n行 {line_num} API错误[{e.status_code}]：{e.message}")
                    return

            except Exception as e:
                print(f"\n行 {line_num} 处理异常：{str(e)}")
                retries += 1
                if retries >= 3:
                    print(f"\n行 {line_num} 重试次数耗尽")

        if retries >= 3 and not self.error_flag.is_set():
            writer.write_line(f"处理失败：{line}")

def process_first_1000_lines(input_path: str, output_path: str, max_workers: int = 100):
    """仅读取前1000行数据，并使用多线程处理"""
    processor = DeepSeekBatchProcessor(max_workers)
    writer = ThreadSafeWriter(output_path)
    batch = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                batch.append( (line_num, line.strip()) )
                if line_num >= 1000:
                    break

        total = len(batch)
        print(f"总数据量：{total} 条")
        processor.process_batch(batch, writer)
        print("\n处理完成！")
    finally:
        writer.close()

if __name__ == '__main__':
    input_file = "data/DSdata.txt"
    output_file = "data/CoTdata.txt"
    process_first_1000_lines(input_file, output_file, max_workers=100)
