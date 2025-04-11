
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

class MyModel():
    def __init__(self):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        lora_path = "DS_RL_model"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = PeftModel.from_pretrained(model, lora_path)
        self.generation_config = {
            "max_new_tokens": 2048,
            "temperature": 0.9,
            "top_p": 1.0,
            "repetition_penalty": 1.2,
        }
    def predict(self, text):
        prompt = "根据以下关键词生成一首歌词，歌词中包含多个句子，句子与句子之间使用/隔开，让我们一步一步的思考（思考过程包含在<think>和</think>之间）：" + text
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(input_ids, **self.generation_config)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return decoded
    #诗，样子，天地：