cpm3_path = "/root/ld/ld_model_pretrained/minicpm3" # if you  want to get react data，cpm3_path can be none
model_path = "/root/ld/ld_model_pretrain/Qwen2.5-72B-Instruct-GPTQ-Int4"  # 教师模型地址
save_question_json = "/root/ld/ld_project/MiniCPM-CookBook/agent_demo/plan_example.json"  # 保存query的json地址
save_complex_question_json = "/root/ld/ld_project/MiniCPM-CookBook/agent_demo/question_complex_react.json"
save_react_qa_json = "/root/ld/ld_project/MiniCPM-CookBook/agent_demo/react_qa_react.json"  # 保存react的json地址
inference_batch_size = 8  # 教师模型生成数据时的batch
gen_datas_per_tool = 10  # 每个tool生成多少条react数据
cpm3_data_save_path = "/root/ld/ld_project/pull_request/MiniCPM_Series_Tutorial/agent_demo/cpm3_fc_train_data.json"  # cpm3的数据保存json地址
complex_example_json = '/root/ld/ld_project/MiniCPM-CookBook/agent_demo/plan_example.json'

params_dict = {
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
