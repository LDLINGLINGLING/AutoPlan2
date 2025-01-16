from tools_call import test_function_call
import copy
from open_api import request_llm,client
from prompt_text import tools_explain,reward_steps_prompt
import re
import json
import json
import concurrent.futures
import threading


def parse_react_data(react_data):
    """
    解析 React 数据并检查其规范性。

    参数：
    - react_data: 字符串形式的 React 数据。

    返回：
    - 包含解析结果的字典，以及数据是否规范的标志。
    """
    import re

    # 初始化结果字典
    result = {
        "question": None,  # 新增字段，用于存储问题
        "steps": [],
        "final_answer": None,
        "is_valid": True,
        "validation_errors": [],
        "error_step": None  # 用于存储出错的步骤编号或 "Final Answer"
    }

    # 正则表达式匹配问题
    question_pattern = re.compile(r"Question:\s*(.*?)\s*(?=Thought:|$)", re.DOTALL)
    question_match = question_pattern.search(react_data)
    if question_match:
        result["question"] = question_match.group(1).strip()
    else:
        result["is_valid"] = False
        result["validation_errors"].append("Question is missing.")  # 添加错误信息

    # 正则表达式匹配 Final Answer
    final_answer_pattern = re.compile(r"Final Answer:\s*(.*?)\s*$", re.DOTALL)
    final_answer_match = final_answer_pattern.search(react_data)
    if final_answer_match:
        result["final_answer"] = final_answer_match.group(1).strip()
    else:
        result["is_valid"] = False
        result["validation_errors"].append("Final Answer is missing.")  # 添加错误信息

    # 分割 react_data 为步骤部分和 Final Answer 部分
    if final_answer_match:
        steps_data = react_data[:final_answer_match.start()].strip()
    else:
        steps_data = react_data.strip()

    # 正则表达式匹配每个步骤
    step_pattern = re.compile(
        r"(Thought:\s*(.*?)\s*)?"
        r"Action:\s*(.*?)\s*"
        r"Action Input:\s*(.*?)\s*"
        r"Observation:\s*(.*?)\s*",
        re.DOTALL  # 允许跨行匹配
    )

    # 查找所有步骤
    steps = list(step_pattern.finditer(steps_data))
    for step in steps:
        thought = step.group(2).strip() if step.group(2) else None
        action = step.group(3).strip()
        action_input = step.group(4).strip()
        observation = step.group(5).strip()

        # 将步骤添加到结果中
        step_data = {
            "Thought": thought,
            "Action": action,
            "Action Input": action_input,
            "Observation": observation
        }

        result["steps"].append(step_data)

    # 检查 Final Answer 是否唯一
    if final_answer_match and "final_answer" in result:
        if len(result["steps"]) > 0:
            result["steps"][-1]["Final Answer"] = result["final_answer"]
            result["final_answer"] = result["steps"][-1]["Final Answer"]
        else:
            result["is_valid"] = False
            result["validation_errors"].append("Final Answer found without any steps.")
            result["error_step"] = "Final Answer"
    else:
        result["is_valid"] = False
        result["validation_errors"].append("Multiple Final Answers found.")
        result["error_step"] = "Final Answer"

    # 检查步骤顺序是否正确
    for i, step in enumerate(result["steps"]):
        if "Thought" not in step or step["Thought"] is None:
            result["is_valid"] = False
            result["validation_errors"].append(f"Step {i + 1} is missing Thought.")
            result["error_step"] = i + 1  # 标记出错的步骤编号
        if "Action" not in step:
            result["is_valid"] = False
            result["validation_errors"].append(f"Step {i + 1} is missing Action.")
            result["error_step"] = i + 1  # 标记出错的步骤编号
        if "Action Input" not in step:
            result["is_valid"] = False
            result["validation_errors"].append(f"Step {i + 1} is missing Action Input.")
            result["error_step"] = i + 1  # 标记出错的步骤编号
        if "Observation" not in step:
            result["is_valid"] = False
            result["validation_errors"].append(f"Step {i + 1} is missing Observation.")
            result["error_step"] = i + 1  # 标记出错的步骤编号

    return result
def steps_dict_to_react_data(result,step=None):
    """
    根据解析后的 result 字典，返回 react_data 前 step 步的所有字符串。
    如果 step 为 None，则返回完整的 react_data。

    参数：
    - result: 解析后的 React 数据字典。
    - step: 需要返回的步数（从 1 开始）。如果为 None，则返回完整的 react_data。

    返回：
    - 前 step 步的完整字符串，或完整的 react_data。
    """
    react_data_up_to_step = []

    # 添加问题
    if result["question"]:
        react_data_up_to_step.append(f"Question: {result['question']}")

    # 如果 step 为 None，返回完整的 react_data
    if step is None:
        for current_step in result["steps"]:
            react_data_up_to_step.append(f"Thought: {current_step['Thought']}")
            react_data_up_to_step.append(f"Action: {current_step['Action']}")
            react_data_up_to_step.append(f"Action Input: {current_step['Action Input']}")
            react_data_up_to_step.append(f"Observation: {current_step['Observation']}")
            # 如果当前步骤包含 Final Answer，则添加 Final Answer
            if "Final Answer" in current_step:
                react_data_up_to_step.append(f"Final Answer: {current_step['Final Answer']}")

        # 如果存在全局的 final_answer，且未被包含在步骤中，则添加到末尾
        if result["final_answer"] and not any("Final Answer" in step for step in result["steps"]):
            react_data_up_to_step.append(f"Final Answer: {result['final_answer']}")

        return "\n".join(react_data_up_to_step)

    # 检查 step 是否有效
    if step < 1 or step > len(result["steps"]):
        raise ValueError(f"Invalid step: {step}. Step must be between 1 and {len(result['steps'])}.")

    # 添加前 step 步的内容
    for i in range(step):
        current_step = result["steps"][i]
        react_data_up_to_step.append(f"Thought: {current_step['Thought']}")
        react_data_up_to_step.append(f"Action: {current_step['Action']}")
        react_data_up_to_step.append(f"Action Input: {current_step['Action Input']}")
        react_data_up_to_step.append(f"Observation: {current_step['Observation']}")

        # 如果当前步骤包含 Final Answer，则添加 Final Answer
        if "Final Answer" in current_step:
            react_data_up_to_step.append(f"Final Answer: {current_step['Final Answer']}")

    # 将列表拼接为字符串
    return "\n".join(react_data_up_to_step)

def get_react_data(data,data_type='alpaca'):
    import re
    assert data_type=='alpaca' or data_type == 'chatml',"data type must be alpaca or chatml"
    if data_type=='alpaca':
        question = 'Question:'+data['input'].split('\n\nQuestion:')[-1]
        react_data = question+'\n'+data['output']
    else:
        question = 'Question:'+data["messages"][1]['content'].split('\n\nQuestion:')[-1]
        react_data = question+'\n'+data["messages"][2]['content']
    return react_data

def extract_reward_dict(text):
    """
    从文本中提取评分字典。

    参数:
        text (str): 输入的文本。

    返回:
        dict: 提取的评分字典。如果未找到匹配的字典，返回 None。
    """
    # 正则表达式
    # pattern = r"\{[^{}]*step\d+:\d+(?:,[^{}]*(?:step\d+:\d+|final_answer:\d+))*[^{}]*\}"
    # match = re.search(pattern, text)
    pattern = r'```json\s*(\{[\s\S]*?\})\s*```'
    # 使用 re.search 查找匹配的字典
    
    matches = list(re.finditer(pattern, text, re.DOTALL))
    try:
        json_str=matches[-1].group(1)
        json_str = re.sub('json\n','',json_str)

        json_dict= json.loads(json_str)
    except:
        json_dict = None
    return json_dict

def get_step_reward(react_data,llm_client):
    """
    reward_dict:{'step1': 5, 'step2': 4, 'final_answer': 5}
    """
    query = tools_explain+react_data+reward_steps_prompt
    try:
        reward_string=request_llm(query,llm_client)
        reward_dict=extract_reward_dict(reward_string)
    except:
        print('the request is error')
        return None
    
    print(reward_dict)
    return reward_dict

def get_reward(reacts_dict,react_data,llm_client):
    before_step_input = steps_dict_to_react_data(reacts_dict)
    reward_steps_dict= get_step_reward(react_data,llm_client)
    if reward_steps_dict == None:
        return None
    steps_dict = copy.deepcopy(reacts_dict['steps'])
    for step,step_dict in enumerate(reacts_dict['steps']):
        try:
            test_function_call(step_dict['Action'],step_dict['Action Input'])
            steps_dict[step]['excute'] = True
            steps_dict[step]['reward'] = reward_steps_dict['step'+str(step+1)]
            #steps_dict[step]['reward'] = get_step_reward(before_step_input,llm_client)
        except Exception as e:
            print(e)
            steps_dict[step]['excute'] = False
            steps_dict[step]['reward'] = reward_steps_dict['step'+str(step+1)]
            #steps_dict[step]['reward'] = get_step_reward(before_step_input,llm_client)
    reacts_dict['steps_reward'] = reward_steps_dict
    reacts_dict['steps'] = copy.deepcopy(steps_dict)
    return reacts_dict

def process_line(line, client):
    react_data = get_react_data(line, data_type='chatml')
    react_dict = parse_react_data(react_data)
    react_dict = get_reward(react_dict, react_data, client)
    if react_dict is None:
        print("the line get the reward is parse error")
        return None
    react_dict['react_data'] = line
    return react_dict

def main(input_json, save_json, client, max_workers=5):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_dict = []
    lock = threading.Lock()  # 创建锁对象

    def safe_append_and_save(react_dict, index):
        with lock:  # 加锁保护共享资源
            all_dict.append(react_dict)
            if index % 5 == 1:
                with open(save_json, "w", encoding="utf-8") as f:
                    json.dump(all_dict, f, ensure_ascii=False, indent=4)
                print(f"save {index} line data in {save_json}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_line = {executor.submit(process_line, line, client): line for line in data[:20]}

        for index, future in enumerate(concurrent.futures.as_completed(future_to_line)):
            line = future_to_line[future]
            try:
                react_dict = future.result()
                if react_dict is not None:
                    safe_append_and_save(react_dict, index)  # 使用线程安全的函数
            except Exception as e:
                print(f"Error processing line: {line}, error: {e}")

    # Final save after all lines are processed
    with lock:  # 加锁保护共享资源
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(all_dict, f, ensure_ascii=False, indent=4)
    print(f"Final save in {save_json}")
if __name__=="__main__":
    input_json = '/DATA/disk0/ld/ld_project/AutoPlan2/train/data/autoplan2/autoplan2_chatml_train.json'
    save_json = "/DATA/disk0/ld/ld_project/AutoPlan2/train/Rewad/autoplan2_train_reward.json"
    main(input_json,save_json,client)