### 必须设置的项
model_path = "/root/ld/ld_model_pretrain/Qwen2.5-72B-Instruct-GPTQ-Int4"  # 教师模型地址
gen_datas_per_tool = 10  # 每个tool生成多少条react数据
params_dict = { # vllm的生成参数
    "n": 1,
    "best_of": 1,
    "presence_penalty": 1,
    "frequency_penalty": 1.0,
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": -1,
    "stop": None,
    "stop_token_ids": None,
    "ignore_eos": False,
    "max_tokens": 4096,
    "logprobs": None,
    "prompt_logprobs": None,
    "skip_special_tokens": True,
}


### 调用get_question函数，获取单链条agent的问题
save_question_json = "AutoPlan2/data_demo/question_react_11_03.json"  # 保存query的json地址

### 调用get_react_data函数，获取简单Agent的训练数据
save_question_json = "AutoPlan2/data_demo/question_react_11_03.json"  # 用于作为react数据的输入
save_react_qa_json = "AutoPlan2/datademo/react_qa_react.json"  # 用于保存单链条Agent训练数据
inference_batch_size = 8  # 教师模型生成数据时的batch

### 调用get_complex_question函数，获取长链条agent数据
save_complex_question_json = "AutoPlan2/datademo/question_complex_react.json" # 用于保存长链条复杂任务的query

### 调用get_complex_react_data函数，获取长链条agent训练数据
save_complex_question_json = "AutoPlan2/datademo/question_complex_react.json" # 用于保存长链条复杂任务的query
complex_example_json = 'AutoPlan2/datademo/plan_example.json' # 用于长链条复杂任务的任务规划示例，请参考示例文件，最少写一个
save_complex_react_qa_json = 'AutoPlan2/data_demo/question_react_11_03.json'
inference_batch_size = 8  # 教师模型生成数据时的batch



