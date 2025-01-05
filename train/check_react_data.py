def parse_react_data(react_data):
    """
    解析 React 数据并检查其规范性。

    参数：
    - react_data: 字符串形式的 React 数据。
    Question: What is the maximum operational range of our helicopter, and does it include the location of the enemy's command center?
    Thought: First, I need to use knowledge_graph to get the helicopter's endurance.
    Action: knowledge_graph
    Action Input: {"weapon_query": "helicopter", "attribute": "endurance"}
    Observation: The helicopter's endurance is: 500km
    Thought: Now that I know the helicopter's endurance is 500km, next, I will use map_search to get the coordinates of the enemy's command center.
    Action: map_search
    Action Input: {"launch": "enemy command center"}
    Observation: {'our helicopter': [100, 80], 'enemy helicopter': [170, 45], 'our command post': [0, 2], 'enemy tank': [20, 13], 'our rocket launcher': [100, 120], 'our firing position 1': [50, 70], 'our firing position 2': [150, 170], 'enemy command center': [70, 35], 'enemy anti-tank missile': [50, 100], 'our tank': [32, 21]}
    Thought: I've obtained the coordinates of the enemy command center, which are [70, 35]. Next, I need to use distance_calculation to find the distance between our helicopter and the enemy command center.
    Action: distance_calculation
    Action Input: {"weapon_query": "{\"our helicopter\":[100,80]}", "map_dict": "{\"enemy command center\":[70,35]}"}
    Observation: Here are all the distances from {'our helicopter': [100, 80]}: {'enemy command center': '54.1km'}
    Thought: I've got the distance between our helicopter and the enemy command center, which is 54.1km. Now I need to compare this distance with the helicopter's endurance.
    Action: python_math
    Action Input: {"math_formulation": "500 > 54.1"}
    Observation: The result is True
    Thought: Since 500km is greater than 54.1km, our helicopter's maximum operational range does include the location of the enemy's command center.
    Final Answer: Our helicopter's maximum operational range includes the location of the enemy's command center.

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

    # 正则表达式匹配每个步骤的完整模式
    step_pattern = re.compile(
        r"Thought:\s*(.*?)\s*"
        r"Action:\s*(.*?)\s*"
        r"Action Input:\s*(.*?)\s*"
        r"Observation:\s*(.*?)\s*"
        r"(?=(Thought:|Final Answer:)|$)",  # 前瞻，确保匹配到下一个 Thought 或 Final Answer
        re.DOTALL  # 允许跨行匹配
    )

    # 正则表达式匹配 Final Answer
    final_answer_pattern = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)

    # 查找所有步骤
    steps = list(step_pattern.finditer(react_data))
    for step in steps:
        thought = step.group(1).strip()
        action = step.group(2).strip()
        action_input = step.group(3).strip()
        observation = step.group(4).strip()

        # 将步骤添加到结果中
        result["steps"].append({
            "Thought": thought,
            "Action": action,
            "Action Input": action_input,
            "Observation": observation
        })

    # 查找 Final Answer
    final_answer_match = final_answer_pattern.search(react_data)
    if final_answer_match:
        result["final_answer"] = final_answer_match.group(1).strip()
    else:
        result["is_valid"] = False
        result["validation_errors"].append("No Final Answer found.")
        result["error_step"] = "Final Answer"  # 标记 Final Answer 出错

    # 检查 Final Answer 是否唯一
    if final_answer_match and len(final_answer_pattern.findall(react_data)) > 1:
        result["is_valid"] = False
        result["validation_errors"].append("Multiple Final Answers found.")
        result["error_step"] = "Final Answer"  # 标记 Final Answer 出错

    # 检查 Final Answer 后面是否还有其他字段
    if final_answer_match:
        remaining_text = react_data[final_answer_match.end():].strip()
        if remaining_text and ("Thought:" in remaining_text or "Action:" in remaining_text or "Observation:" in remaining_text):
            result["is_valid"] = False
            result["validation_errors"].append("Fields found after Final Answer.")
            result["error_step"] = "Final Answer"  # 标记 Final Answer 出错

    # 检查步骤顺序是否正确
    for i, step in enumerate(result["steps"]):
        if "Thought" not in step:
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
