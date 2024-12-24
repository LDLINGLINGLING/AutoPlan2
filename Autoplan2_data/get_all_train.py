import json
import re

import re

def get_tag_indices(input_string):
    """
    获取字符串中所有特定标签（Thought, Action, Action Input, Observation, Final Answer）的索引。
    
    参数:
    input_string (str): 输入字符串
    
    返回:
    dict: 各标签的索引位置
    """
    tags = ["Thought", "Action", "Action Input", "Observation", "Final Answer"]
    tag_indices = {tag: [] for tag in tags}
    
    for tag in tags:
        # 使用正则表达式查找所有标签的起始索引
        indices = [match.start() for match in re.finditer(tag, input_string)]
        tag_indices[tag] = indices
    
    return tag_indices

def truncate_after_final_answer(input_string):
    """
    根据"Final Answer"后最近的标签截断字符串。
    如果"Final Answer"不存在、出现多次，或不是最晚出现的标签，返回None。
    
    参数:
    input_string (str): 输入字符串
    
    返回:
    str: 截断后的字符串，如果有问题则返回None
    """
    # 获取标签的索引
    tag_indices = get_tag_indices(input_string)
    
    # 获取"Final Answer"标签的所有索引
    final_answer_indices = tag_indices["Final Answer"]
    
    # 1. 如果没有"Final Answer"，返回None
    if not final_answer_indices:
        return None
    
    # 2. 如果"Final Answer"出现超过一次，返回None
    if len(final_answer_indices) > 1:
        return None
    
    # 3. 如果"Final Answer"不是最晚出现的标签，返回None
    max_final_answer_index = final_answer_indices[0]
    for tag, indices in tag_indices.items():
        if tag != "Final Answer" and indices:
            if max(indices) > max_final_answer_index:
                return None
    
    # 如果通过了上述检查，继续执行截断
    closest_next_index = len(input_string)  # 默认是字符串末尾
    for tag in tag_indices:
        if tag != "Final Answer":
            for index in tag_indices[tag]:
                if index > max_final_answer_index and index < closest_next_index:
                    closest_next_index = index
    
    # 截断字符串
    truncated_string = input_string[:closest_next_index]
    return truncated_string
train_json = ['/root/ld/ld_project/AutoPlan2/Autoplan2_data/complex_react_qa_11_09_o1.json','/root/ld/ld_project/AutoPlan2/Autoplan2_data/complex_react_qa_11_09.json','/root/ld/ld_project/AutoPlan2/Autoplan2_data/save_complex_qa_11_11.json']
all_train = []
for file in train_json:
    with open(file, 'r') as file:
        data_string = file.read()
    split_list = data_string.split('###')
    json_list=[]
    for i in split_list:
        try:
            json_list.append(json.loads(i))
        except:
            continue
    all_train.extend(json_list)
all_train_data = []
for i in all_train:
    try:
        if truncate_after_final_answer(i['output']) !=None:
            i['input'] = i['input'].split('请按以下执行顺序完成以上问题，请注意A、B、C等等都是指代某些特殊值。：')[0]
            i['output'] = truncate_after_final_answer(i['output'])
            all_train_data.append(i)
    except:
        continue
with open('/root/ld/ld_project/AutoPlan2/Autoplan2_data/all_train.json', 'w', encoding='utf-8') as file:
    json.dump(all_train_data, file, ensure_ascii=False, indent=4)