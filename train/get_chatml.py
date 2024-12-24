# 转换为 ChatML 格式
import os
import shutil
import json

input_dir = "/root/ld/ld_project/MiniCPM/finetune/data/ocnli_public"
output_dir = "/root/ld/ld_project/MiniCPM/finetune/data/ocnli_public_chatml"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

for fn in ["train.json", "dev.json"]:
    data_out_list = []
    with open(os.path.join(input_dir, fn), "r") as f, open(os.path.join(output_dir, fn), "w") as fo:
        for line in f:
            if len(line.strip()) > 0:
                data = json.loads(line)
                data_out = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"请判断下边两个句子的关系属于 [entailment, neutral, contradiction]中的哪一种？\n句子1: {data['sentence1']}\n句子2：{data['sentence2']}\n"
                        },
                        {
                            "role": "assistant",
                            "content": data["label"],
                        },
                    ]
                }
                data_out_list.append(data_out)
        json.dump(data_out_list, fo, ensure_ascii=False, indent=4)
