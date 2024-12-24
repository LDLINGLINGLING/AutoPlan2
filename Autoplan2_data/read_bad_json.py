import json
with open('/root/ld/ld_project/AutoPlan2/Autoplan2_data/save_complex_qa_11_11.json','r',encoding='utf-8') as f:
    jsonstr=f.read()
data_list=jsonstr.split('###')
print(data_list[3])

json_list =[]
for i in data_list:
    try:
        json_list.append(json.loads(i))
    except:
        continue
print(len(json_list))
with open('/root/ld/ld_project/AutoPlan2/Autoplan2_data/complex_react_qa_11_11.json','w',encoding='utf-8') as f:
    json.dump(json_list,f)
