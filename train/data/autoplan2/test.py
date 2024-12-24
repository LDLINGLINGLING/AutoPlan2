import json
import random

# 假设有一个名为data.json的文件
with open('/root/ld/ld_project/MiniCPM/finetune/data/autoplan2/autoplan2_chatml_test.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
print(data)