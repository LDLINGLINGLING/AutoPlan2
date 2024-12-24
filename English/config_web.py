import gradio as gr
from main import *

# 定义基础参数
default_model_path = "/root/ld/ld_model_pretrain/Qwen2.5-72B-Instruct-GPTQ-Int4"
default_inference_batch_size = 8
default_tensor_parallel_size = 1

# 实际函数定义
def interface(choice, model_path, model_api, inference_batch_size, tensor_parallel_size, **kwargs):
    # 根据模型类型选择正确的模型输入
    model = model_path if model_type.value == "本地模型" else model_api
    
    # 如果选择了API模型，则设置inference_batch_size为None
    if model_type.value == "API模型":
        inference_batch_size = None
    
    if choice == "简单Agent问题构造":
        return get_question(model, kwargs['save_question_json'], inference_batch_size)
    elif choice == "简单Agent训练数据构造":
        return get_react_data(model, kwargs['input_question_json'], kwargs['save_react_qa_json'], inference_batch_size)
    elif choice == "复杂Agent 问题构造":
        return get_complex_question(model, kwargs['save_complex_question_json'], kwargs['complex_example_json'], inference_batch_size)
    elif choice == "复杂Agent 训练数据构造":
        if model_type.value == "API模型":
            return get_complex_react_data(model, kwargs['input_complex_question_json'], kwargs['save_complex_react_qa_json'], inference_batch_size, kwargs['except_comple_react_json'])
        else:
            return get_complex_react_data(model, kwargs['input_complex_question_json'], kwargs['save_complex_react_qa_json'], inference_batch_size, kwargs['except_comple_react_json'])

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# Agent Data Generation Interface")
    
    # 模型选择
    with gr.Row():
        model_type = gr.Radio(choices=["本地模型", "API模型"], label="选择模型类型")
        model_path_input = gr.Textbox(value=default_model_path, label="本地模型路径 (仅当选择本地模型时有效)", visible=True)
        model_api_input = gr.Textbox(value="", label="API模型配置 (仅当选择API模型时有效)", visible=False)
        tensor_parallel_size_input = gr.Number(value=default_tensor_parallel_size, label="Num of tensor parallel (仅当选择本地模型时有效)", visible=True)

    # 推理批次大小
    inference_batch_size_input = gr.Number(value=default_inference_batch_size, label="Inference Batch Size (仅当选择本地模型时有效)", visible=True)

    # 功能选择
    function_choice = gr.Dropdown(choices=["简单Agent问题构造", "简单Agent训练数据构造", "复杂Agent 问题构造", "复杂Agent 训练数据构造"], label="选择功能")

    # 简单Agent问题构造
    with gr.Column(visible=False) as simple_question_inputs:
        save_question_json_input = gr.Textbox(value="/root/ld/ld_project/AutoPlan2/data_demo/question_react.json", label="保存问题JSON路径")

    # 简单Agent训练数据构造
    with gr.Column(visible=False) as simple_react_data_inputs:
        input_question_json_input = gr.Textbox(value="/root/ld/ld_project/AutoPlan2/data_demo/question_react_11_06.json", label="用于React数据的输入问题JSON")
        save_react_qa_json_input = gr.Textbox(value="/root/ld/ld_project/AutoPlan2/data_demo/react_qa_react_11_06.json", label="保存React QA JSON路径")

    # 复杂Agent问题构造
    with gr.Column(visible=False) as complex_question_inputs:
        save_complex_question_json_input = gr.Textbox(value="/root/ld/ld_project/AutoPlan2/data_demo/question_complex_react.json", label="保存复杂问题JSON路径")
        complex_example_json_input = gr.Textbox(value='/root/ld/ld_project/AutoPlan2/data_demo/plan_example.json', label="复杂任务示例JSON路径")

    # 复杂Agent训练数据构造
    with gr.Column(visible=False) as complex_react_data_inputs:
        input_complex_question_json_input = gr.Textbox(value="/root/ld/ld_project/AutoPlan2/data_demo/plan_example.json", label="用于复杂React数据的输入问题JSON")
        save_complex_react_qa_json_input = gr.Textbox(value='/root/ld/ld_project/AutoPlan2/data_demo/complex_react_qa_11_07.json', label="保存复杂React QA JSON路径")
        except_comple_react_json_input = gr.Textbox(value='/root/ld/ld_project/AutoPlan2/data_demo/except_react_11_07.json', label="异常处理React JSON路径")

    # 输出
    output = gr.Textbox(label="输出")

    # 提交按钮
    btn = gr.Button("生成数据")

    # 按钮点击事件
    btn.click(
        fn=interface,
        inputs=[
            function_choice,
            model_path_input,
            model_api_input,
            inference_batch_size_input,
            tensor_parallel_size_input,
            save_question_json_input,
            input_question_json_input,
            save_react_qa_json_input,
            save_complex_question_json_input,
            complex_example_json_input,
            input_complex_question_json_input,
            save_complex_react_qa_json_input,
            except_comple_react_json_input
        ],
        outputs=output
    )

    # 更新可见性
    def update_visibility(choice, model_type):
        visibility = {
            "simple_question_inputs": False,
            "simple_react_data_inputs": False,
            "complex_question_inputs": False,
            "complex_react_data_inputs": False,
            "model_path_input": model_type == "本地模型",
            "model_api_input": model_type == "API模型",
            "inference_batch_size_input": model_type == "本地模型",
            "tensor_parallel_size_input": model_type == "本地模型"
        }

        if choice == "简单Agent问题构造":
            visibility["simple_question_inputs"] = True
        elif choice == "简单Agent训练数据构造":
            visibility["simple_react_data_inputs"] = True
        elif choice == "复杂Agent 问题构造":
            visibility["complex_question_inputs"] = True
        elif choice == "复杂Agent 训练数据构造":
            visibility["complex_react_data_inputs"] = True

        return [
            gr.update(visible=visibility["model_path_input"]),
            gr.update(visible=visibility["model_api_input"]),
            gr.update(visible=visibility["inference_batch_size_input"]),
            gr.update(visible=visibility["tensor_parallel_size_input"]),
            gr.update(visible=visibility["simple_question_inputs"]),
            gr.update(visible=visibility["simple_react_data_inputs"]),
            gr.update(visible=visibility["complex_question_inputs"]),
            gr.update(visible=visibility["complex_react_data_inputs"])
        ]

    # 当功能选择或模型类型改变时更新可见性
    function_choice.change(
        fn=update_visibility,
        inputs=[function_choice, model_type],
        outputs=[
            model_path_input,
            model_api_input,
            inference_batch_size_input,
            tensor_parallel_size_input,
            simple_question_inputs,
            simple_react_data_inputs,
            complex_question_inputs,
            complex_react_data_inputs
        ]
    )
    model_type.change(
        fn=update_visibility,
        inputs=[function_choice, model_type],
        outputs=[
            model_path_input,
            model_api_input,
            inference_batch_size_input,
            tensor_parallel_size_input,
            simple_question_inputs,
            simple_react_data_inputs,
            complex_question_inputs,
            complex_react_data_inputs
        ]
    )

# 启动Gradio应用
demo.launch()
