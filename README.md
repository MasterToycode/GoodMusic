# 歌词生成系统

AIGC创意文本生成，使用LoRA和强化学习进行微调。
注意：Data数据文件太大了，DS_data.txt以及processed_data.txt文件您下载之后还需要解压。如果您想直接下载原始数据文件，请去我的HuggingFace主页下载原始数据

## 项目结构

### 代码文件
- `code/__main__.py`: 主程序入口，启动GUI界面
- `code/_MyModel.py`: 核心模型实现，加载DeepSeek/Qwen模型和LoRA适配器
- `code/UI.py`: PyQt5实现的用户界面
- `code/reward.py`: 强化学习的奖励函数实现
- `code/GRPO.ipynb`: 基于规则的策略优化训练流程
- `code/data_process.py`: 数据处理脚本
- `code/LORA.py`: LoRA模型实现
- `code/LORA_with_CoT.py`: 带思维链的LoRA实现

### 数据文件夹
- `data/`: 存放训练数据(CoTdata.txt, DSdata.txt等)
- `data/CoTdata.txt`: 带思维链的训练数据
- `data/DSdata.txt`: 关键词：原文训练数据
- `data/processed_data.txt`: 处理后的训练数据

### 模型文件夹
- `DS_LoRA/`: 基础DeepSeek模型的LoRA适配器
- `DS_RL_model/`: 强化学习微调的DeepSeek模型 
- `Qwen_LoRA/`: 基础Qwen模型的LoRA适配器
- `Qwen_CoT_LoRA/`: 带思维链的基础Qwen模型适配器

## 使用方法

1. 安装依赖:
- (请根据您的GPU cuda版本下载对应的torch版本！)
- conda create -n Goodmusic python==3.11
- conda activate Goodmusic
- pip install -r requirements.txt


3. 运行程序:
python code/__main__.py

4. 在GUI界面输入关键词，生成歌词

## 模型训练

1. 数据准备: 将训练数据放入data/文件夹
2. 运行GRPO.ipynb进行模型训练
3. 训练好的模型会保存在对应模型文件夹
