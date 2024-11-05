# AutoPlan系列发展概述

## AutoPlan1

### 背景与挑战
- 在AutoPlan1推出之前，市场上的智能代理（Agents）大多功能较为基础，面对复杂的问题处理能力有限。
- 这些早期的Agents缺乏处理复杂逻辑或执行长序列任务的能力，或者即便能够尝试解决，其解决方案的准确性也常常不尽如人意。

### 目标与成果
- **核心目的**：AutoPlan1旨在克服上述局限性，通过开发出能够胜任特定领域内更复杂工作的高级Agents。
- **技术进步**：
  - 引入了先进的算法和技术框架，显著提升了Agents在处理细分领域复杂任务时的表现。
  - 提高了Agent系统的稳定性以及解决问题的准确度。
  - 成功地从仅能执行简单任务的初级Agents过渡到了具备强大处理能力的高级Agents。

## AutoPlan2

### 新挑战与改进方向
- 尽管AutoPlan1成功地提高了Agents应对复杂场景的能力，但这一过程中也暴露出了一些新的问题。
  - 构建这类复杂Agent的成本较高，尤其是涉及到大量人工标注的数据准备阶段。
  - 对于那些需要快速部署新类型Agent的应用场景来说，高昂的成本成为了阻碍因素之一。

### 目标与创新
- **核心目标**：AutoPlan2项目致力于降低新Agent的创建门槛，特别是针对那些要求迅速上线且性能稳定的场合。
- **关键技术点**：
  - **冷启动优化**：开发了一套机制使得即使是初次接触某一领域的Agent也能基于现有知识库快速形成初步服务能力。
  - **成本控制**：通过自动化工具减少对人工干预的需求，在保证至少90%服务准确性的同时，力求将复杂Agent的数据准备成本控制在总成本的10%以内。
  - **兼容性增强**：确保即使是为简单任务设计的Agent也能无缝接入更加复杂的系统中使用，无需额外调整。
# 效果展示：
## [零人工构造长链条数据](./data_demo)
### 以下数据为500条中随意挑选，未经任何人工修改或者挑选。
```plantext
Question1:
确认我方直升机的最大作战范围内是否包含敌方的指挥中心位置。

Plan:
1.使用knowledge_graph获取直升机的续航里程A。
2.使用map_search获取敌方指挥中心的坐标B。
3.使用distance_calculation计算我方直升机和敌方指挥中心的距离C。
4.对比续航里程A和距离C，确认是否在作战范围内。

React_data:
Thought: 首先，我需要使用knowledge_graph来获取直升机的续航里程。
Action: knowledge_graph
Action Input: {\"weapon_query\": \"直升机\", \"attribute\": \"续航\"}\nObservation: 直升机的续航是:500km
Thought: 现在我知道了直升机的续航里程是500km，接下来我需要使用map_search来获取敌方指挥中心的坐标。
Action: map_search\nAction Input: {\"lauch\": \"敌指挥中心\"}
Observation: {'我方直升机': [100, 80], '敌直升机': [170, 45], '我方指挥所': [0, 2], '敌坦克': [20, 13], '我方火箭炮': [100, 120], '我方发射阵地1': [50, 70], '我方发射阵地2': [150, 170], '敌指挥中心': [70, 35], '敌反坦克导弹': [50, 100], '我方坦克': [32, 21]} 
Thought: 我已经获取到了敌方指挥中心的坐标是[70, 35]，接下来我需要使用distance_calculation来计算我方直升机和敌方指挥中心的距离。
Action: distance_calculation\nAction Input: {\"weapon_query\": \"{\\\"我方直升机\\\":[100,80]}\", \"map_dict\": \"{\\\"敌指挥中心\\\":[70,35]}\"}\nObservation: 以下是所有单位与{'我方直升机': [100, 80]}的距离:{'敌指挥中心': '54.1km'} 
Thought: 我已经得到了我方直升机和敌方指挥中心的距离是54.1km，现在需要对比这个距离与直升机的续航里程。
Action: python_math\nAction Input: {\"math_formulation\": \"500 > 54.1\"}
Observation: 执行结果是True\nThought: 由于500km大于54.1km，所以我方直升机的最大作战范围内包含敌方的指挥中心位置。
Final Answer: 我方直升机的最大作战范围内包含敌方的指挥中心位置。
```
```plantext
Question2:请选择武器打击敌方坦克，打击后将结果图片给到张三将军.

Plan:
1.使用map_search获取敌方坦克的位置坐标A
2.使用knowledge_graph查询敌方坦克的克制武器B
3.使用weapon_launch发射克制武器B打击敌方坦克位置坐标A
4.使用map_search查询克制武器B的位置坐标C
5.使用knowledge_graph获取克制武器B的速度D
6.使用distance_calculation计算克制武器B的位置坐标C到敌方坦克的位置坐标A的距离E
7.根据距离E和速度D计算武器飞行时间F
8.使用Situation_display显示敌方坦克位置坐标A,获得图片G
9.使用address_book查询张三将军的QQ_EmailH。
10.F时间后使用QQ_Email将图片C发送到张三将军的QQ_EmailH

React_data:
Thought: 首先，我需要使用map_search来获取敌方坦克的位置坐标A。
Action: map_search
Action Input: {\"lauch\": \"敌坦克\"}
Observation: {'我方直升机': [100, 80], '敌直升机': [170, 45], '我方指挥所': [0, 2], '敌坦克': [20, 13], '我方火箭炮': [100, 120], '我方发射阵地1': [50, 70], '我方发射阵地2': [150, 170], '敌指挥中心': [70, 35], '敌反坦克导弹': [50, 100], '我方坦克': [32, 21]} 
Thought: 我已经获取到了敌方坦克的位置坐标A，接下来需要使用knowledge_graph查询克制敌方坦克的武器B。
Action: knowledge_graph\nAction Input: {\"weapon_query\": \"敌坦克\", \"attribute\": \"克制武器\"}
Observation: '豹2A7坦克': { '克制武器': '反坦克导弹'} 
Thought: 我已经获取到了克制敌方坦克的武器B，接下来需要使用weapon_launch发射克制武器B打击敌方坦克位置坐标A。
Action: weapon_launch
Action Input: {\"weapon_query\": \"反坦克导弹\", \"target_name\": \"敌坦克\", \"target_coordinate\": \"[20, 13]\"}
Observation: 已启动反坦克导弹打击敌坦克打击位置：[20, 13] \nThought: 武器已经发射，接下来需要使用map_search查询克制武器B的位置坐标C。
Action: map_search
Action Input: {\"lauch\": \"反坦克导弹\"}
Observation: { '敌反坦克导弹': [50, 100]} 
Thought: 我已经获取到了克制武器B的位置坐标C，接下来需要使用knowledge_graph获取克制武器B的速度D。
Action: knowledge_graph
Action Input: {\"weapon_query\": \"反坦克导弹\", \"attribute\": \"速度\"}
Observation: 反坦克导弹的速度是:200km/h 
Thought: 我已经获取到了克制武器B的速度D，接下来需要使用distance_calculation计算克制武器B的位置坐标C到敌方坦克的位置坐标A的距离E。
Action: distance_calculation
Action Input: {\"weapon_query\": \"{'反坦克导弹':[50, 100]}\", \"map_dict\": \"{'敌坦克': [20, 13]}\"}
Observation: 以下是所有单位与{'反坦克导弹': [50, 100]}的距离:{'敌坦克': '92.0km'} 
Thought: 我已经获取到了克制武器B的位置坐标C到敌方坦克的位置坐标A的距离E，接下来需要根据距离E和速度D计算武器飞行时间F。
Action: python_math
Action Input: {\"math_formulation\": \"92.0/200\"}
Observation: 执行结果是0.46 
Thought: 我已经计算出了武器飞行时间F，接下来需要使用Situation_display显示敌方坦克位置坐标A,获得图片G。
Action: Situation_display
Action Input: {\"coordinate\": \"[20, 13]\", \"radio\": \"300\"}\nObservation: 已经显示以[20, 13]为中心以300为半径的态势地图,图片地址为/ai/ld/picture1.png 
Thought: 我已经获取到了图片G，接下来需要使用address_book查询张三将军的QQ_EmailH。
Action: address_book
Action Input: {\"person_name\": \"张三\"}
Observation: {'邮箱': '45123456@qq.com', '电话': '13713156111', '部队': '黑狐坦克', '职务': '副总师'} 
Thought: 我已经获取到了张三将军的QQ_EmailH，接下来需要使用QQ_Email将图片C发送到张三将军的QQ_EmailH。
Action: QQ_Email
Action Input: {\"E-mail_address\": \"45123456@qq.com\", \"E-mail_content\": \"打击敌方坦克的结果如图所示：/ai/ld/picture1.png\"}
Observation: 已将打击敌方坦克的结果如图所示：/ai/ld/picture1.png发送到45123456@qq.com \nThought: 我已经完成了所有任务。
Final Answer: 通过以上步骤，我成功使用反坦克导弹打击了敌方坦克
```

# 使用方法
1. 首先修改config.py配置文件,[config.py](./config.py)文件中已有注释。
2. 修改tools_caption.py文件，填写Agent需要调用工具的配置（可多个）。
```python
tools=[
  {
        "name_for_human": "知识图谱", # name_for_human写工具名称，用来给模型提示，需要能够望文生义，
        "name_for_model": "knowledge_graph", # name_for_model是函数名称，是真实调用工具的函数名
        "excute_function": True, # excute_function表示构造数据时，是否真实的调用工具以获得更加高质量的数据。
        "description_for_model": "知识图谱是输入武器种类获取该武器的属性，也可以输入某种属性获得所有武器的该属性",# description_for_model是用于提示模型，本工具的作用。需要描述清晰明了
        "example": "帮我查一下敌方直升机的续航里程", # example 表示一个能使用本工具解决的问题。
        "parameters": [
            {
                "name": "weapon_query",# 这是第一个参数名称
                "description": "武器名称", # 这是对本参数的文字描述，用于提示模型，需要能够望文生义
                "scope": ["直升机", '豹2A7坦克','黑狐坦克',"雷达",'装甲车','狙击枪', "反坦克导弹", "直升机", "火箭炮", "所有武器",'无人机','步兵'],  # 参数的取值范围
                "required": True, # 这个参数是否是必须的
                "schema": {"type": "string"}, # 本参数的数据结构类型
            },
            {
                "name": "attribute",# 这是第二个参数名称
                "description": "武器的属性",# 这是对本参数的文字描述，用于提示模型，需要能够望文生义
                "scope": ['射程','携带武器','重量','速度','适应场景','克制武器','续航','满载人数','飞行高度','承载重量','所有属性','探测范围'],# 参数的取值范围
                "required": True,# 这个参数是否是必须的
                "schema": {"type": "string"},# 本参数的数据结构类型
            },
        ],
    }
    ]
```
3. 对于第二步中"excute_function": True的工具，需要在tools_call.py的文件中进行实现。
```python
if plugin_name == "knowledge_graph":
        weapon_name = args_dict["weapon_query"] # 输入参数的解析，需要和第步中参数的name值相同(weapon_query)
        attribute = args_dict["attribute"] # 输入参数的解析，需要和第步中参数的name值相同(attribute)
        kg = {
            "直升机": {
                "飞行高度": "0.3km以内",
                "携带武器": "火箭弹",
                "克制武器": "对空导弹",
                "重量": "3000kg",
                "速度": "100km/h",
                "射程": "2km",
                "适应场景": "空战",
                "续航": "500km",
                "满载人数": "7人",
                "承载重量": "10000kg",
                "续航里程": "1000km",
            },
            #、、、、、、、、、、此处省略
            "狙击枪": {"射程": "1.2km", "重量": "30kg", "适应场景": "暗杀"},
        }
        if weapon_name != "所有武器":
            try:
                return "{}的{}是:{}".format(
                    weapon_name, attribute, kg[weapon_name][attribute]
                )
            except:
                return kg
        return kg #返回值需要是容易被模型理解的文字
```
4. 根据具体需求，修改main.py文件最下方，执行以下函数的其中一个,执行前请保证第一步根据执行函数进行了配置：
  ```python
  if __name__ == "__main__":
      #get_question()# 获取简单Agent问题数据
      #get_complex_question()
      #get_react_data # 获取agent执行过程数据
      get_complex_react_data() # 获取复杂agent的执行过程数据
  ```
5. 执行get_react_data()或get_complex_react_data()后保存的训练数据已经保存为llamafactory的qwen训练格式，可以直接进行sft训练。
