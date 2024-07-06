
import torch
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

# 导入环境
df = pd.read_json('/root/autodl-tmp/cyp/data/medical.train2.json')
ds = Dataset.from_pandas(df)

from modelscope import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen1.5-7B-Chat",
    device_map=device
)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# model.print_trainable_parameters()
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen1.5-7B-Chat")

def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"system\n你现在是一个实体识别模型，你需要识别文本里面的实体，如果存在结果，返回实体类型。\nuser\n{example['text']}\nassistant\n", add_special_tokens=False)
    response = tokenizer(f"{example['label']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# # 创建模型
# import torch

# model = AutoModelForCausalLM.from_pretrained('qwen/Qwen1.5-7B-Chat', device_map="auto", torch_dtype=torch.bfloat16)
# model.eval()
# model.print_trainable_parameters()

# lora
from peft import LoraConfig, TaskType, get_peft_model,PeftModel

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True, # 预测模式
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = PeftModel.from_pretrained(model, '/root/autodl-tmp/output/Qwen1.5/checkpoint-5300/', torch_dtype=torch.float32, trust_remote_code=True)
# # model = get_peft_model(model, config)
model.half().cuda()
model.eval()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# 准备预测数据
text =  "医学实体识别：\n目的观察复方丁香开胃贴外敷神阙穴治疗慢性心功能不全伴功能性消化不良的临床疗效\n实体识别选项：中医治则 ,中医治疗 ,中医证候 ,中医诊断 ,中药 ,临床表现 ,其他治疗 ,方剂 ,西医治疗 ,西医诊断\n 答：", # 修改为你想要预测的输入文本
# instruction = tokenizer([text], add_special_tokens=False)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# 生成预测
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=512)

# 解码预测结果
predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(predicted_text)