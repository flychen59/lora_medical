import os
import json

def process_text(path):
    text = ''
    result_dict = {}

    with open(path, 'r') as file:
        lines = file.readlines()
        current_string = ""
        messages = []
        for i, line in enumerate(lines):
            entity_sentence = ""
            if line.strip():  # 如果行不为空
                word = line.split(' ')[0]
                text += word

                index = line.find('B')
                if index != -1:
                    label = line.strip().split('-')[1]
                if line.find('B') != -1 or line.find('I') != -1:
                    current_string += word

                if (line.find('O') != -1 and current_string != '') or (i+1 < len(lines) and lines[i+1] == '\n'):
                    current_string = current_string.strip()
                    if current_string:  # 确保current_string不为空
                        if label in result_dict:
                            if current_string not in result_dict[label]:
                                result_dict[label].append(current_string)
                        else:
                            result_dict[label] = [current_string]
                    current_string = ''
                       
            else:
                for key in result_dict:
                    result_dict[key] = list(filter(lambda x: x, set(result_dict[key])))  # 去掉空字符串并去重
                    
                for key, value in result_dict.items():
                    value=''.join(value)
                    if value == "":
                        entity_sentence = "没有找到任何实体"
                    entity_sentence += f"""{{"entity_text": "{str(value)}", "entity_label": "{key}"}}"""
                   
                message = {
                "instruction": """你是一个医学实体识别领域的专家，你需要从给定的句子中提取中医治则 ,中医治疗 ,中医证候 ,中医诊断 ,中药 ,临床表现 ,其他治疗 ,方剂 ,西医治疗 ,西医诊断. 以 json 格式输出, 如 {"entity_text": "口苦", "entity_label": "临床表现"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". """,
                "input": f"文本:{text}",
                "output": entity_sentence,
            }
            
                messages.append(message)
                
                
                
                text = ''
                result_dict = {}

    return messages

if __name__ == '__main__':
    path = '/root/autodl-tmp/data_process/medical.dev'
    messages = process_text(path)
    path_json = path.replace(os.path.basename(path), os.path.basename(path) + '4.json')
    # json.dump(
    #     result,
    #     open(path_json, "w", encoding="utf-8"),
    #     ensure_ascii=False,
    #     indent=2
    # )
     # 保存重构后的JSONL文件
    with open(path_json, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


