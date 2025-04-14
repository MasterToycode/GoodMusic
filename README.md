# 歌词生成系统 - 使用LoRA和强化学习的AIGC创意文本生成

## 项目简介
本项目是一个基于DeepSeek和Qwen大语言模型的歌词生成系统，通过LoRA微调和强化学习技术优化生成质量。系统提供GUI界面，用户输入关键词即可生成创意歌词。

## 系统要求
- Python 3.11
- CUDA 12.6 (如需GPU加速)
- 至少16GB内存 (推荐32GB)
- 支持PyTorch的NVIDIA GPU (推荐)

## 安装指南

### 1. 创建conda环境
```bash
conda create -n Goodmusic python=3.11
conda activate Goodmusic
```

### 2. 安装PyTorch
根据您的硬件选择以下命令之一：

**CUDA 12.6版本:**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**CPU版本:**
```bash
pip3 install torch torchvision torchaudio
```

### 3. 安装项目依赖
```bash
pip install -r requirements.txt
```

## 配置说明

### 模型配置
系统支持以下预训练模型适配器：
- `DS_LoRA/`: 基础DeepSeek模型的LoRA适配器
- `DS_RL_model/`: 强化学习微调的DeepSeek模型
- `Qwen_LoRA/`: 基础Qwen模型的LoRA适配器
- `Qwen_CoT_LoRA/`: 带思维链的基础Qwen模型适配器

默认使用DS_RL_model，如需切换模型，请修改`code/_MyModel.py`中的`model_path`参数。

### 推理参数配置
可在`code/_MyModel.py`中调整以下参数：
- `max_length`: 生成文本最大长度
- `temperature`: 生成随机性
- `top_p`: 核采样参数
- `repetition_penalty`: 重复惩罚系数

## 使用方法

### 启动GUI界面
```bash
python code/__main__.py
```

### 界面操作指南
1. 在输入框输入关键词(如"爱情"、"夏天")
2. 点击"生成"按钮
3. 等待生成结果(CPU推理可能需要较长时间)



## 模型训练

### 数据准备
1. 将训练数据放入`data/`文件夹
2. 支持的数据格式：
   - `CoTdata.txt`: 带思维链的训练数据
   - `DSdata.txt`: 关键词-原文对训练数据
   - `processed_data.txt`: 预处理后的训练数据

### 训练流程
1. **LoRA微调**:
   ```bash
   python code/LORA.py
   ```
   或带思维链版本:
   ```bash
   python code/LORA_with_CoT.py
   ```

2. **强化学习优化**:
   运行`code/GRPO.ipynb`笔记本进行基于规则的策略优化

### 训练参数调整
各训练脚本中提供详细参数注释，主要可调整：
- 学习率
- 训练轮次
- 批大小
- LoRA秩参数

## 项目结构
```
project/
├── README.md              # 项目说明文档
├── requirements.txt       # Python依赖包列表
│
├── code/                  # 源代码目录
│   ├── __main__.py        # 主程序入口(GUI启动)
│   ├── _MyModel.py        # 核心模型实现(加载/推理)
│   ├── UI.py              # PyQt5用户界面实现
│   ├── reward.py          # 强化学习奖励函数定义
│   ├── data_process.py    # 数据预处理和清洗
│   ├── deepseek_vaule.py  # 调用DeepSeek模型评价相关工具
│   ├── getCOT.py          # 思维链生成工具
│   ├── GRPO.ipynb         # 强化学习训练笔记本
│   ├── LORA.py            # LoRA微调基础实现
│   ├── LORA_with_CoT.py   # 带思维链的LoRA微调
│   ├── test.ipynb         # 测试和实验笔记本
│   └── threads_data_extract.py  # 多线程数据处理得到关键词对数据s
│
├── data/                  # 数据目录
│   ├── CoTdata.txt        # 带思维链的训练数据
│   ├── DSdata.txt         # 原始训练数据(关键词-歌词对)
│   └── processed_data.txt # 预处理后的训练数据
│
├── DS_LoRA/               # DeepSeek基础LoRA适配器
│
├── DS_RL_model/           # 强化学习微调模型
│
├── Qwen_LoRA/             # Qwen基础LoRA适配器
│
└── Qwen_CoT_LoRA/         # 带思维链的Qwen适配器
```

## 常见问题
1. **GPU内存不足**:
   - 减小`max_length`参数
   - 使用`LORA.py`中的`fp16`选项

2. **生成质量不佳**:
   - 调整`temperature`和`top_p`参数
   - 使用带思维链的模型(Qwen_CoT_LoRA)

3. **安装问题**:
   - 确保Python版本为3.11
   - 检查CUDA/cuDNN版本匹配
