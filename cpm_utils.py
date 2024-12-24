import json
def save_cpm3_data(cpm3_data_path, cpm3_data):
    # 将列表转换为 JSON 格式的字符串
    json_str = json.dumps(cpm3_data, ensure_ascii=False, indent=4)

    # 将 JSON 字符串保存到文件
    with open(cpm3_data_path, "w", encoding="utf-8") as json_file:
        json_file.write(json_str)


def switch_cpm_tool(tools):
    format_tool = {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID.",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    }
    cpm_tools = []
    for tool in tools:
        format_tool["function"]["name"] = tool["name_for_model"]
        format_tool["function"]["description"] = tool["description_for_model"]
        # format_tool['function']["parameters"]['properties']=
        required_list = []
        for param in tool["parameters"]:
            """param{
                    'name': 'weapon_query',
                    'description': '武器名称',
                    'scope':['直升机','坦克','反坦克导弹','直升机','火箭炮','所有武器'],
                    'required': True,
                    'schema': {'type': 'string'},
                }"""
            format_tool["function"]["parameters"]["properties"][param["name"]] = {
                "type": param["schema"]["type"],
                "description": param["description"],
            }
            if param["required"]:
                required_list.append(param["name"])
        format_tool["function"]["parameters"]["required"] = required_list
        format_tool["function"]["parameters"]["additionalProperties"] = False
        cpm_tools.append(format_tool)
    return cpm_tools


def get_cpm_function_call():
    with open(save_react_qa_json, "r", encoding="utf-8") as file:
        # 将json文件内容解析为Python对象
        react_qa = json.load(file)
    cpm_tool = switch_cpm_tool(tools)
    tokenizer = AutoTokenizer.from_pretrained(cpm3_path, trust_remote_code=True)
    cpm_fc_train_data = []
    for react in react_qa:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
            }
        ]
        query = react["input"].split("Question: ")[-1]
        print(query)
        react_str = list(react.values())[-1]
        Thought1, Action, Action_Input, Observation, Thought2, Final_Answer = split_react_data(
            react_str
        )
        if (
            Thought1
            and Action
            and Action_Input
            and Observation
            and Thought2
            and Final_Answer
        ):
            messages.append({"role": "user", "content": query})
            prompt = tokenizer.apply_chat_template(
                messages, tools=cpm_tool, tokenize=False, add_generation_prompt=True
            )
            cpm_thought1 = "<|thought_start|>\n{}\n<|thought_end|>".format(Thought1)
            cpm_function_and_param = "\n<|tool_call_start|>\n```python\n{}({})\n```\n<|tool_call_end|>".format(
                Action, re.sub(": ", "=", Action_Input)
            )

            cpm_fc_train_data.append(
                [
                    {"role": "system", "content": prompt.split("<|im_end|>")[0][19:]},
                    {"role": "user", "content": query},
                    {
                        "role": "assistant",
                        "content": cpm_thought1 + cpm_function_and_param,
                    },
                ]
            )

            cpm_response = "<|im_end|>\n<|im_start|>tool\n{}<|im_end|>\n<|im_start|>assistant\n".format(
                Observation
            )
            cpm_thought2 = "<|thought_start|>\n{}\n<|thought_end|>\n".format(Thought2)
            cpm_answer = Final_Answer
            cpm_fc_train_data.append(
                [
                    {"role": "system", "content": prompt.split("<|im_end|>")[0][19:]},
                    {
                        "role": "user",
                        "content": query
                        + "<|im_start|>assistant\n"
                        + cpm_function_and_param
                        + cpm_response,
                    },
                    {"role": "assistant", "content": cpm_thought2 + cpm_answer},
                ]
            )
        else:
            print(1)
            continue
        # cpm_fc_train_data.append({"role":"system",'content':prompt+cpm_function_and_param+cpm_response,'role':'assistant','content':cpm_thought2+cpm_answer})
    save_cpm3_data(cpm3_data_save_path, cpm_fc_train_data)
    print(
        "{}条cpm3 function call数据已经保存到{}".format(
            len(cpm_fc_train_data), cpm3_data_save_path
        )
    )