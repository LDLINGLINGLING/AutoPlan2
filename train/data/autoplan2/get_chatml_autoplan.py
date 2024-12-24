import json
import random

# 假设有一个名为data.json的文件
with open('/root/ld/ld_project/AutoPlan2/Autoplan2_data/all_train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

chatml_data = []
bad_case = 0
for d in data:
    if '执行失败' not in d['output']:
        chatml_data.append({"messages":[{   "role":"system","content": "'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'"},
            {"role":"user","content":d['input'] },
            {"role":"assistant", "content": d['output']
        }]})
    else:
        bad_case += 1

print('bad_case',bad_case)
random.shuffle(chatml_data)
with open('/root/ld/ld_project/MiniCPM/finetune/data/autoplan2/autoplan2_chatml_train.json', 'w', encoding='utf-8') as file:
    json.dump(chatml_data[:-200], file, ensure_ascii=False, indent=4)
with open('/root/ld/ld_project/MiniCPM/finetune/data/autoplan2/autoplan2_chatml_test.json', 'w', encoding='utf-8') as file:
    json.dump(chatml_data[-200:], file, ensure_ascii=False, indent=4)