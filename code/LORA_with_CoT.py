import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

base_model  = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载数据
raw_data_path = r"data/CoTdata.txt"  
with open(raw_data_path, "r", encoding="utf-8") as f:
    raw_lines = f.readlines()

# 处理每一行数据，解析出关键词、思维链和诗歌内容
def process_line(line):
    # 使用 [：:] 同时匹配中文和英文冒号
    pattern = r"^(.*?)<think>(.*?)</think>[：:](.*)$"
    match = re.match(pattern, line.strip())
    if match:
        keywords = match.group(1).strip()
        cot = match.group(2).strip()
        poem = match.group(3).strip()
        # 构造训练实例：输入部分给出提示和关键词，输出部分包含完整思维链及答案
        training_text = (
            f"【输入】：根据以下关键词生成一首歌词，歌词中包含多个句子，确保句子通顺、诗意、格式正确。"
            f"让我们一步一步的思考（思考过程包含在<think>和</think>之间）：{keywords}\n\n"
            f"【输出】：<think>{cot}</think>\n{poem}"
        )
        return training_text
    else:
        # 如果格式不符，输出提示并返回 None
        print("跳过格式错误的行：", line.strip())
        return None

# 解析所有数据行
processed_samples = []
for line in raw_lines:
    result = process_line(line)
    if result:
        processed_samples.append(result)

# 构建 Hugging Face 数据集
dataset = Dataset.from_dict({"text": processed_samples})

# 加载基础模型和 LoRA 模型
model = PeftModel.from_pretrained(base_model, r"D:\GoodMusicV3.0\3_24_LoRA").to("cuda")  # 替换为你的 LoRA 路径
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=8,                  # 低秩矩阵的秩，常取 8、16 或 32
    lora_alpha=32,        # 缩放因子，控制 LoRA 影响
    target_modules=["q_proj", "k_proj", "v_proj"],  # 应用 LoRA 的模块，通常是注意力层的投影
    lora_dropout=0.1,     # Dropout 概率，防止过拟合
    bias="none",       # 通常设为 "none"
    task_type="CAUSAL_LM" 
)
model = get_peft_model(model, lora_config)
model.cuda()

# 分词函数：对文本进行分词，并构造 labels
def tokenize_function(examples):
    # 此处的文本已经包含了输入和输出的完整内容
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 对数据集进行映射处理
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./lora",
    num_train_epochs=1000,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10000,
    save_steps=15000,
    fp16=True,
)

# 构造 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()

# 推理示例
generation_config = {
    "max_new_tokens": 1024,
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.2,
    "do_sample": True,
    "encoder_no_repeat_ngram_size": 4,
}
if True:
    prompt = "根据以下关键词生成一首歌词，歌词中包含多个句子，句子与句子之间使用/隔开，让我们一步一步的思考（思考过程包含在<think>和</think>之间）：温柔，轮廓，洒脱："
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids, **generation_config)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

print(decoded)

# 保存模型
model.save_pretrained("4_2_LoRA_3")
