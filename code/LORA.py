from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

raw_data_path = ""#替换为对应的数据集路径  
with open(raw_data_path, "r", encoding="utf-8") as f:
    raw_lines = f.readlines()

def process_line(line):
    segments = line.strip().split("/")
    return "/".join(segments[:-1]) if len(segments) > 1 else line.strip()

processed_samples = [process_line(line) for line in raw_lines if line.strip()]
dataset = Dataset.from_dict({"text": processed_samples})

model_name = ""#替换为对应的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,                  # 低秩矩阵的秩，通常取 8、16 或 32
    lora_alpha=32,         # 缩放因子，控制 LoRA 的影响
    target_modules=["q_proj", "v_proj"],  # 应用 LoRA 的模块，通常是注意力层的投影
    lora_dropout=0.1,      # Dropout 概率，防止过拟合
    bias="none",           # 是否训练偏置，通常设为 "none"
    task_type="CAUSAL_LM"  # 任务类型，对于因果语言模型使用 "CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

def tokenize_function(examples):
    # 预定义固定的提示词
    prompt = "根据以下关键词生成一首歌词，歌词中包含多个句子，句子与句子之间使用/隔开，让我们一步一步的思考（思考过程包含在<think>和</think>之间）："

    # 在原文本前面加上提示词
    modified_texts = [prompt + text for text in examples["text"]]

    # 进行分词
    tokenized = tokenizer(modified_texts, truncation=True, padding="max_length", max_length=256)

    # 复制 input_ids 作为 labels
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized




tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./lora",
    num_train_epochs=8,
    per_device_train_batch_size=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10000,
    save_steps=15000,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)


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

model.save_pretrained("")#替换为对应的保存路径