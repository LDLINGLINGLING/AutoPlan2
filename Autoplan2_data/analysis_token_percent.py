from transformers import AutoTokenizer
import numpy as np
import json
# 假设你的tokenizer位于'model_name_or_path'
model_name_or_path = '/root/ld/ld_model_pretrain/Qwen2.5-3B-Instruct'  # 替换成你的tokenizer的实际路径或名称
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
with open('/root/ld/ld_project/AutoPlan2/Autoplan2_data/all_train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
# 示例字符串列表
texts = [i['instruction']+i['input']+i['output'] for i in data]

# 使用tokenizer计算每个字符串的token数量
token_counts = [len(tokenizer.tokenize(text)) for text in texts]

# 计算分位点
percentiles = [80, 90, 95, 97, 98, 99, 100]
results = np.percentile(token_counts, percentiles)

# 输出结果
for percentile, result in zip(percentiles, results):
    print(f"{percentile}th percentile: {result:.2f} tokens")