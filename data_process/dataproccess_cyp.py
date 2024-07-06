import os
import json

def process_text(path):
    text = ''
    result = []
    result_dict = {}
    label_list = []

    label_path = '/root/autodl-tmp/cyp/data/labels.txt'
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            index = line.find('B')
            if index != -1:
                label = line.strip().split('-')[1]
                if label not in label_list:
                    label_list.append(label)

    with open(path, 'r') as file:
        lines = file.readlines()
        current_string = ""
        for i, line in enumerate(lines):
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
                result.append({
                    "text": "医学实体识别：\n" + text + "\n实体识别选项：" + ' ,'.join(label_list) + '\n 答：',
                    "label": '上述句子中的实体包含：' + ''.join(f'\n {key}实体：{value}' for key, value in result_dict.items()),
                })
                
                
                
                
                
                text = ''
                result_dict = {}
                
                
                
                

    return result

if __name__ == '__main__':
    path = '/root/autodl-tmp/cyp/data/medical.train'
    result = process_text(path)
    path_json = path.replace(os.path.basename(path), os.path.basename(path) + '3.json')
    json.dump(
        result,
        open(path_json, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2
    )




    with open(path, 'r') as file:
        lines = file.readlines()
        current_string = ""
        for i, line in enumerate(lines):
            if line.strip():  # 如果行不为空
                word = line.split(' ')[0]
                text += word