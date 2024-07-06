import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from seqeval.metrics import f1_score,classification_report
import pandas as pd
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# Metric
def compute_metrics(response,gt_value):
        # preds, labels = eval_preds
        # if isinstance(preds, tuple):
        #     preds = preds[0]
        # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # if data_args.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(response, gt_value):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            hypothesis = ' '.join(hypothesis)
            if not hypothesis:
                hypothesis = "-"
            scores = rouge.get_scores(hypothesis, ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

# 加载原下载路径的tokenizer
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/qwen/Qwen2-1.5B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/qwen/Qwen2-1.5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)


# 加载训练好的Lora模型，将下面的checkpoint-[XXX]替换为实际的checkpoint文件名名称
model = PeftModel.from_pretrained(model, model_id="/root/autodl-tmp/output/Qwen2-NER_medical_num50/checkpoint-7500")


test_jsonl_new_path='/root/autodl-tmp/data_process/medical.test4.json'
test_df=test_df = pd.read_json(test_jsonl_new_path, lines=True)

all_true_labels = []
all_pred_labels = []
for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    gt_value=row['output']
    
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": f"{response}"})
    compute_metrics(response,gt_value)
    
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    all_true_labels.append([gt_value])
    all_pred_labels.append([messages[2]['content']])
    # if gt_value['entity_text']!=messages[2]['content']['entity_text']:
    #     print(gt_value)
    # #     print(all_pred_labels)
    # import ast
    # dictionary = ast.literal_eval(gt_value)
   
    # nums=0
    # for i in range(len(all_true_labels)):
    #     if all_true_labels[i]!=all_pred_labels[i]:
    #         print('all_true_labels{} \n all_pred_labels{}'.format(all_true_labels[i],all_pred_labels[i]))
    #         nums+=1
        
    #     if messages[2]['content'].startswith("'") and messages[2]['content'].endswith("'"):
    #             result = messages[2]['content'][1:-1]
    
f1_dev = f1_score(all_true_labels, all_pred_labels)

detailed_results=classification_report(all_true_labels, all_pred_labels)
# Print detailed classification report
print(detailed_results)

# Display detailed results
detailed_results_df = pd.DataFrame(detailed_results)
print(detailed_results_df)
print(f1_dev)
 





# input_text='柴胡疏肝散对功能性消化不良患者胃动力及胃肠激素的影响'
# test_texts ={
# "instruction":"""你是一个医学实体识别领域的专家，你需要从给定的句子中提取中医治则 ,中医治疗 ,中医证候 ,中医诊断 ,中药 ,临床表现 ,其他治疗 ,方剂 ,西医治疗 ,西医诊断. 以 json 格式输出, 如 {"entity_text": "口苦", "entity_label": "临床表现"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". """,
# "input": f"文本:{input_text}"
# }


# instruction = test_texts['instruction']
# input_value = test_texts['input']

# messages = [
#     {"role": "system", "content": f"{instruction}"},
#     {"role": "user", "content": f"{input_value}"}
# ]

# response = predict(messages, model, tokenizer)
# print(response)