import re

def contains_chinese(text):
    """
    Unicode 范围 \u4e00-\u9fff 包含常见的汉字
    """
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def process_lyrics(text):
    """
    处理歌词文本：
      1. 按 '/' 分割
      2. 去除空白及空行
      3. 过滤掉不包含中文（视为英文）的歌词
      4. 去除重复歌词（保持原始顺序）
    """
    # 使用 '/' 分割字符串得到歌词列表
    lyrics = text.split('/')
    processed = []
    seen = set()
    
    for line in lyrics:
        # 去除两端空白
        line = line.strip()
        # 如果为空则跳过
        if not line:
            continue
        # 如果这句歌词不包含中文，则视为英文歌词，跳过
        if not contains_chinese(line):
            continue
        if len(line) < 3:
            continue
        # 去重：如果该句未出现过，则添加到结果中
        if line not in seen:
            seen.add(line)
            processed.append(line)
    
    return processed

def main():
    input_filename = 'data\lyrics.txt'
    output_filename = 'data\processed_data.txt'
    
    # 读取原始数据文件，建议使用 utf-8 编码
    with open(input_filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 处理歌词数据
    processed = process_lyrics(content)
    
    # 处理后的数据以 '/' 重新拼接，也可以改成每行一个
    output_content = '/'.join(processed)
    
    # 将处理后的数据写入输出文件
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f'处理完成，结果保存在 {output_filename}')


if __name__ == '__main__':
    main()
