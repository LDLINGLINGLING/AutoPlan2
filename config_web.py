import gradio as gr
import re
from vllm import LLM, SamplingParams
from config import *
from build_react_prompt import (
    build_input_text,
    TOOL_DESC,
    PROMPT_REACT,
    parse_latest_plugin_call,
)
import os
from tools_call import function_call
import json
import json5
from transformers import AutoTokenizer
from tools_caption import tools
from cpm_utils import save_cpm3_data,switch_cpm_tool
# from agent_demo import *

tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code = True)
# 定义基础参数
model_path = "/root/ld/ld_model_pretrain/Qwen2.5-72B-Instruct-GPTQ-Int4"
gen_datas_per_tool = 10
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
sampling_params = SamplingParams(**params_dict)
def split_react_data(react_str):
    pattern = re.compile(
        r"Thought:\s*(.*?)\nAction:\s*(.*?)\nAction Input:\s*(.*?)\nObservation:\s*(.*?)\nThought:\s*(.*?)\nFinal Answer:\s*(.*)",
        re.DOTALL,
    )

    matches = pattern.findall(react_str)
    try:
        for match in matches:
            Thought1 = match[0]
            Action = match[1]
            Action_Input = match[2]
            Observation = match[3]
            Thought2 = match[4]
            Final_Answer = match[5]
        return Thought1, Action, Action_Input, Observation, Thought2, Final_Answer
    except:
        return None, None, None, None, None, None


def get_answer_from_output(output):
    pattern = r"「问题开始」(.*?)「问题结束」"
    questions = re.findall(pattern, output, re.DOTALL)
    questions = [q.strip() for q in questions]
    return questions

def get_complex_question_from_output(output):
    pattern = r"--复杂问题_start--(.*?)--复杂问题_end--"
    questions = re.findall(pattern, output, re.DOTALL)
    pattern2 = r"--任务规划_start--(.*?)--任务规划_end--"
    task_plan = re.findall(pattern2, output, re.DOTALL)
    if len(task_plan) == len(questions):
        questions = [{q.strip():task_plan[i]} for i,q in enumerate(questions)]
    else:
        questions = []
    return questions


def get_tool_description(tool):
    tool_descp = "工具名称是{},作用是{},".format(
        tool["name_for_model"], tool["description_for_model"]
    )
    for t in tool["parameters"]:
        if t["required"]:
            if t["scope"]:
                tool_descp += "参数“{}”是必须输入的，作用是{},该参数的取值范围是{}。".format(
                    t["name"], t["description"], t["scope"]
                )
            else:
                tool_descp += "参数“{}”是必须输入的，作用是{}。".format(t["name"], t["description"])
        elif t["scope"]:
            tool_descp += "参数“{}”是可选的，作用是{},该参数的取值范围是{}。".format(
                t["name"], t["description"], t["scope"]
            )
        else:
            tool_descp += "参数“{}”是可选的，作用是{}。".format(t["name"], t["description"])
    return tool_descp


def get_question():
    if 'llm' not in locals():
        llm = LLM(
            model=model_path,
            tensor_parallel_size=8,
            max_model_len=4096,
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
        )

    prompt_template = """你是一个智能助手，现在我请你为以下工具生成问题，要求生成的问题能够被这个工具解决。工具的详细介绍如下：\n{}\n我现在给你一个关于此工具问题的示例「问题开始」
    {}「问题结束」,接下来请你根据此示例和工具描述再生成{}个能够使用该工具解决的问题，并且用「问题开始」和「问题结束」将其包裹,不要生成其他的无关文字。"""

    all_questions = []
    all_react_prompt = []
    questinos_dict = {}
    for tool in tools:
        questions = []
        while True:
            tool_description = get_tool_description(tool)
            input_prompt = prompt_template.format(
                tool_description, tool["example"], gen_datas_per_tool
            )
            input_prompt = """<|im_start|> system\n you are a helpful assistant<|im_end|>\n<|im_start|> user\n {}<|im_end|>\n<|im_start|> assistant\n""".format(
                input_prompt
            )
            outputs = llm.generate(input_prompt, sampling_params)
            output = outputs[0].outputs[0].text
            questions.extend(get_answer_from_output(output))

            if len(questions) >= gen_datas_per_tool:
                all_questions.extend(questions)
                print(questions)
                questinos_dict[tool["name_for_model"]] = questions
                break

    with open(save_question_json, "w", encoding="utf-8") as f:
        json.dump(questinos_dict, f, ensure_ascii=False, indent=4)
        print("{}条输入指令已经保存到{}".format(len(all_questions), save_question_json))

def get_complex_question():
    if 'llm' not in locals():
        llm = LLM(
            model=model_path,
            tensor_parallel_size=8,
            max_model_len=4096,
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
        )
    if complex_example_json.endswith('json'):
        with open(complex_example_json,'r',encoding='utf-8') as f:
            examples = json.load(f)
    else:
        examples = []
    all_questions = []
    questions = []
    for example in examples:
        question,plan = list(example.items())[0]
        example = '''--复杂问题_start--\n{}\n--复杂问题_end--\n--任务规划_start--\n{}\n--任务规划_end--'''.format(question,plan)
        example += """--复杂问题2_start--
        计算一下，如果敌方有意向我方发射阵地1的300人使用化学武器，我方用一架直升机需要多久可以疏散所有人员到我方指挥所完成？
        --复杂问题2_end--
        --任务规划2_start--
        1使用knowledge_graph获取我们直升机的承载人数A.
        2根据我放发射阵地1的300人和直升机的承载人数B计算需要的疏散次数C.
        3使用knowledge_graph获取直升机的速度D.
        4使用map_search获取我方指挥所的坐标F和我方发射阵地1的位置G.
        5使用distance_calculation计算F到G的距离H.
        6计算直升机以速度D行驶距离H来回一次需要的时间I.
        7计算完成C次疏散需要的总时间J，并判断是否在直升机续航E之内.
        --任务规划2_end--
        """
        prompt_template = """\n所有工具的详细介绍如上所示，你是一个智能助手，现在我请你为以下工具生成复合问题，要求生成的问题是具体的问题,有具体的目标，实体，任务，且能够被这几个工具所解决，并且最少这个复杂问题最少要使用两个以上的工具才能完成。接下来请你根据每个工具的简单问题示例和工具描述再生成{}个能够使用以上最少两个工具解决的复杂问题及其解决方案，接下来我会给你一个示例：/n{}，\n.并且严格按照示例的格式,用--复杂问题_start--和--复杂问题_end--以及--任务规划_start--和--任务规划_end--进行包裹，请以「任务规划_end」结束，不要生成其他无关的文字."""
        tool_prompt = ''
        questinos_dict = {}
        for index,tool in enumerate(tools):
            questions = []

            tool_description = get_tool_description(tool)
            tool_prompt += '\n第{}个工具：'.format(index+1)+tool_description
        input_prompt = tool_prompt + prompt_template.format(gen_datas_per_tool,example) 
        messages = [
            {"role": "system", "content":"You are a helpful assistant."},
            {"role": "user", "content": input_prompt}
        ]
        input_prompt= tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # input_prompt = """<|im_start|> system\n You are a helpful assistant<|im_end|>\n<|im_start|> user\n {}<|im_end|>\n<|im_start|> assistant\n""".format(
        #     input_prompt
        # )
        while True:
            outputs = llm.generate(input_prompt, sampling_params)
            output = outputs[0].outputs[0].text
            print(output)
            questions.extend(get_complex_question_from_output(output))

            if len(questions) >= gen_datas_per_tool:
                break
        with open(save_complex_question_json, "w", encoding="utf-8") as f:
            json.dump(questinos_dict, f, ensure_ascii=False, indent=4)
            print("{}条输入指令已经保存到{}".format(len(all_questions), save_complex_question_json))
def get_react_data():
    if 'llm' not in locals():
        llm = LLM(
            model=model_path,
            tensor_parallel_size=8,
            max_model_len=4096,
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
        )

    with open(save_question_json, "r", encoding="utf-8") as file:
        # 将json文件内容解析为Python对象
        tool_questions = json.load(file)
    all_questions = []
    for i in tool_questions:
        all_questions.extend(tool_questions[i])
    react_question = [build_input_text([(q, "")], tools) for q in all_questions]
    params_dict["top_k"] = 1
    params_dict["stop"] = ["Observation:"]

    
    sampling_params = SamplingParams(**params_dict)
    for index in range(0, len(react_question), inference_batch_size):
        react_qa = []
        outputs = llm.generate(
            react_question[index : index + inference_batch_size], sampling_params
        )
        for i in range(len(outputs)):
            output = outputs[i].outputs[0].text
            
            plugin_name, plugin_args, text = parse_latest_plugin_call(output)
            excute_flag = True
            for tool in tools:
                print(tool)
                if (
                    tool["name_for_model"] == plugin_name
                    and tool["excute_function"] == False
                ):
                    excute_flag = False
                    second_input = (
                        react_question[index + i] + output + "Observation: "
                    )
                    output2 = (
                        llm.generate(second_input, sampling_params)[0]
                        .outputs[0]
                        .text
                    )
            if excute_flag:
                observation = function_call(plugin_name, plugin_args,llm,tokenizer)
                second_input = (
                    react_question[index + i]
                    + output
                    + "Observation: {}".format(observation)
                )
            output2 = llm.generate(second_input, sampling_params)[0].outputs[0].text
            print(output2)
            # react_qa.append({react_question[index+i]: second_input[len(react_question[index+i]):]+output2})
            react_qa.append(
                {
                    "instruction": "You are a helpful assistant.",
                    "input": react_question[index + i][75:-33],
                    "output": second_input[len(react_question[index + i]) :]
                    + output2,
                }
            )
            with open(save_react_qa_json, "a+", encoding="utf-8") as f:
                json.dump(react_qa, f, ensure_ascii=False, indent=4)
                print("{}条react qa数据已经保存到{}".format(len(react_qa), save_react_qa_json))
def get_complex_react_data():
    if 'llm' not in locals():
        llm = LLM(
            model=model_path,
            tensor_parallel_size=8,
            max_model_len=4096,
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
        )

    with open(save_complex_question_json ,"r", encoding="utf-8") as file:
        # 将json文件内容解析为Python对象
        tool_questions = json.load(file)
    all_questions = []
    for i in tool_questions:
        all_questions.append(''.join([f"{key}请按以下执行顺序完成以上问题，请注意A、B、C等等都是指代某些特殊值。：{value}" for key, value in i.items()]))
    react_question = [build_input_text([(q, "")], tools) for q in all_questions]
    params_dict["top_k"] = 1
    params_dict["stop"] = ["Observation:"]

    
    sampling_params = SamplingParams(**params_dict)
    for index in range(0, len(react_question), inference_batch_size):
        react_qa = []
        outputs = llm.generate(
            react_question[index : index + inference_batch_size], sampling_params
        )
        for i in range(len(outputs)):
            output = outputs[i].outputs[0].text
            all_output = react_question[index+i] + output
            try:
                while 'Final Answer' not in output:
                    plugin_name, plugin_args, text = parse_latest_plugin_call(output)
                    excute_flag = True
                    for tool in tools:
                        if (
                            tool["name_for_model"] == plugin_name
                            and tool["excute_function"] == False
                        ):
                            excute_flag = False
                            second_input = (
                                all_output + "Observation: "
                            )
                            output = (
                                llm.generate(second_input, sampling_params)[0]
                                .outputs[0]
                                .text
                            )
                            all_output += "Observation: " + output
                            
                    if excute_flag:
                        observation = function_call(plugin_name, plugin_args,llm,tokenizer)
                        second_input = (
                            all_output
                            + "Observation: {}".format(observation)
                        )
                        output = llm.generate(second_input, sampling_params)[0].outputs[0].text
                        print(observation)
                        print(output)
                        all_output += "Observation: {}".format(observation)+output
                    # react_qa.append({react_question[index+i]: second_input[len(react_question[index+i]):]+output2})
                react_qa.append(
                    {
                        "instruction": "You are a helpful assistant.",
                        "input": react_question[index + i][75:-33],
                        "output": all_output[len(react_question[index + i]):]
                    }
                )
            except:
                print('#####{}#####'.format(all_questions[index+i]))
                continue

        with open(save_complex_react_qa_json, "a+", encoding="utf-8") as f:
            json.dump(react_qa, f, ensure_ascii=False, indent=4)
            print("{}条react qa数据已经保存到{}".format(len(react_qa), save_react_qa_json))


# Gradio界面定义
def interface(choice, model_path, save_question_json, save_react_qa_json, save_complex_question_json, complex_example_json, inference_batch_size,save_complex_react_qa_json):
    if choice == "简单Agent问题构造":
        return get_question(save_question_json)
    elif choice == "简单Agent训练数据构造":
        return get_react_data(save_question_json, save_react_qa_json, inference_batch_size)
    elif choice == "复杂Agent 问题构造":
        return get_complex_question(save_complex_question_json)
    elif choice == "复杂Agent 训练数据构造":
        return get_complex_react_data(save_complex_question_json, complex_example_json, save_complex_react_qa_json, inference_batch_size)

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# Agent Data Generation Interface")
    
    # 基础选项
    with gr.Row():
        model_path_input = gr.Textbox(value=model_path, label="Model Path (Required)")
    
    # 函数选择
    with gr.Row():
        function_choice = gr.Dropdown(choices=["简单Agent问题构造", "简单Agent训练数据构造", "复杂Agent 问题构造", "复杂Agent 训练数据构造"], label="Select Function")

    # 根据选择的函数显示相关输入
    with gr.Column(visible=False) as question_inputs:
        save_question_json_input = gr.Textbox(value="AutoPlan2/data_demo/question_react_11_03.json", label="get_question时作为保存路径，get_react_data时作为react数据的输入")

    with gr.Column(visible=False) as react_data_inputs:
        save_react_qa_json_input = gr.Textbox(value="AutoPlan2/datademo/react_qa_react.json", label="get_react_data单链条React训练数据保存地址")
        inference_batch_size_input = gr.Number(value=8, label="Inference Batch Size ")

    with gr.Column(visible=False) as complex_question_inputs:
        save_complex_question_json_input = gr.Textbox(value="AutoPlan2/datademo/question_complex_react.json", label="get_complex_question时作为保存问题地址，get_complex_react_data时作为query输入")

    with gr.Column(visible=False) as complex_react_data_inputs:
        complex_example_json_input = gr.Textbox(value='AutoPlan2/datademo/plan_example.json', label="复杂问题和任务规划的示例json")
        save_complex_react_qa_json_input = gr.Textbox(value='AutoPlan2/data_demo/question_react_11_03.json', label="保存复杂react训练数据 ")

    # 输出组件
    output = gr.Textbox(label="Output")

    # 创建提交按钮
    btn = gr.Button("Generate Data")

    # 设置按钮点击时的行为
    btn.click(
        fn=interface,
        inputs=[
            function_choice,
            model_path_input,
            save_question_json_input,
            save_react_qa_json_input,
            save_complex_question_json_input,
            complex_example_json_input,
            inference_batch_size_input
        ],
        outputs=output
    )

    # 根据选择的函数切换可见性
    def update_visibility(choice):
        visibility = {
            "question_inputs": True,
            "react_data_inputs": False,
            "complex_question_inputs": False,
            "complex_react_data_inputs": False
        }

        if choice == "简单Agent问题构造":
            visibility["question_inputs"] = True
        elif choice == "简单Agent训练数据构造":
            visibility["question_inputs"] = True
            visibility["react_data_inputs"] = True
        elif choice == "复杂Agent 问题构造":
            visibility["complex_question_inputs"] = True
        elif choice == "复杂Agent 训练数据构造":
            visibility["complex_question_inputs"] = True
            visibility["complex_react_data_inputs"] = True
            visibility["react_data_inputs"] = True

        return [
            gr.update(visible=visibility["question_inputs"]),
            gr.update(visible=visibility["react_data_inputs"]),
            gr.update(visible=visibility["complex_question_inputs"]),
            gr.update(visible=visibility["complex_react_data_inputs"])
        ]

    function_choice.change(
        fn=update_visibility,
        inputs=function_choice,
        outputs=[
            question_inputs,
            react_data_inputs,
            complex_question_inputs,
            complex_react_data_inputs
        ]
    )

# 启动Gradio应用
demo.launch()