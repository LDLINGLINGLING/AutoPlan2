import json
import re
with open('/root/ld/ld_project/MiniCPM-CookBook/agent_demo/train_plan.json','r',encoding='utf-8') as f:
    example_list = json.load(f)
new_example_list = []
for i in example_list:
    key = re.sub('\nassistant\nThought:','',i["conversations"][0]['value'].split('Begin!\n\nQuestion: ')[-1])
    value = re.sub('\nAction:','',i["conversations"][1]['value'])
    if len(value.split("\n")) > 2:
        new_example_list.append({key:value})

with open('/root/ld/ld_project/MiniCPM-CookBook/agent_demo/plan_example.json','w',encoding='utf-8') as f:
    json.dump(new_example_list, f, ensure_ascii=False, indent=4)