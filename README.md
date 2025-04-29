
# AutoPlan2
## [中文](./readme_1.md)
## Download Data
[please click here the password is adan](https://pan.baidu.com/s/1SUxJY4zMMTVeGxcoq2UuQQ?pwd=adan)
## [AutoPlan1](https://github.com/LDLINGLINGLING/AutoPlan)
### Background and Challenges
- Before the launch of AutoPlan1, most intelligent agents (Agents) on the market had relatively basic functionalities with limited capability to handle complex problems.
- These early Agents lacked the ability to process complex logic or perform long sequences of tasks, or even if they could attempt to solve them, the accuracy of their solutions was often unsatisfactory.

### Goals and Achievements
- **Core Purpose**: AutoPlan1 aims to overcome these limitations by developing advanced Agents capable of performing more complex work within specific domains.
- **Technological Advancements**:
  - The introduction of advanced algorithms and technical frameworks has significantly improved the performance of Agents in handling complex tasks within specialized areas.
  - The stability of Agent systems and the accuracy of problem-solving have been enhanced.
  - A successful transition from basic Agents that can only perform simple tasks to advanced Agents with powerful processing capabilities has been achieved.

## AutoPlan2

### New Challenges and Improvement Directions
- Although AutoPlan1 successfully enhanced the Agents' ability to cope with complex scenarios, this process also exposed some new issues.
  - The cost of building such complex Agents is high, especially during the data preparation phase, which involves extensive manual labeling.
  - For application scenarios requiring rapid deployment of new types of Agents, the high costs became a hindrance.

### Goals and Innovations
- **Core Goal**: The AutoPlan2 project is dedicated to lowering the entry barrier for creating new Agents, particularly for those situations demanding swift deployment with stable performance.
- **Key Technical Points**:
  - **Cold Start Optimization**: Developed a mechanism allowing Agents, even when first encountering a particular domain, to quickly establish initial service capabilities based on existing knowledge bases.
  - **Cost Control**: By employing automation tools, the need for human intervention is reduced, aiming to maintain at least 90% service accuracy while striving to limit the data preparation costs for complex Agents to no more than 10% of the total cost.
  - **Enhanced Compatibility**: Ensures that Agents designed for simple tasks can be seamlessly integrated into more complex systems without additional adjustments.

# Effect Demonstration:
## [Zero Manual Construction of Long COT Data](./data_demo)
### The following data is randomly selected from 500 entries, without any manual modification or selection.
```plaintext
Question1:
Confirm whether the maximum operational range of our helicopter includes the location of the enemy's command center.

Plan:
1. Use knowledge_graph to obtain the helicopter's endurance A.
2. Use map_search to get the coordinates B of the enemy's command center.
3. Use distance_calculation to calculate the distance C between our helicopter and the enemy's command center.
4. Compare the endurance A with the distance C to confirm if it is within the operational range.

React_data:
Thought: First, I need to use knowledge_graph to get the helicopter's endurance.
Action: knowledge_graph
Action Input: {\"weapon_query\": \"helicopter\", \"attribute\": \"endurance\"}
Observation: The helicopter's endurance is: 500km
Thought: Now that I know the helicopter's endurance is 500km, next, I will use map_search to get the coordinates of the enemy's command center.
Action: map_search
Action Input: {\"launch\": \"enemy command center\"}
Observation: {'our helicopter': [100, 80], 'enemy helicopter': [170, 45], 'our command post': [0, 2], 'enemy tank': [20, 13], 'our rocket launcher': [100, 120], 'our firing position 1': [50, 70], 'our firing position 2': [150, 170], 'enemy command center': [70, 35], 'enemy anti-tank missile': [50, 100], 'our tank': [32, 21]}
Thought: I've obtained the coordinates of the enemy command center, which are [70, 35]. Next, I need to use distance_calculation to find the distance between our helicopter and the enemy command center.
Action: distance_calculation
Action Input: {\"weapon_query\": \"{\\\"our helicopter\\\":[100,80]}\", \"map_dict\": \"{\\\"enemy command center\\\":[70,35]}\"}
Observation: Here are all the distances from {'our helicopter': [100, 80]}: {'enemy command center': '54.1km'}
Thought: I've got the distance between our helicopter and the enemy command center, which is 54.1km. Now I need to compare this distance with the helicopter's endurance.
Action: python_math
Action Input: {\"math_formulation\": \"500 > 54.1\"}
Observation: The result is True
Thought: Since 500km is greater than 54.1km, our helicopter's maximum operational range does include the location of the enemy's command center.
Final Answer: Our helicopter's maximum operational range includes the location of the enemy's command center.
```
```plaintext
Question2: Select a weapon to strike the enemy tank and send the result image to General Zhang San.

Plan:
1. Use map_search to get the location coordinates A of the enemy tank.
2. Use knowledge_graph to query the counter-weapon B for the enemy tank.
3. Use weapon_launch to fire the counter-weapon B at the enemy tank's location coordinates A.
4. Use map_search to query the location coordinates C of the counter-weapon B.
5. Use knowledge_graph to get the speed D of the counter-weapon B.
6. Use distance_calculation to calculate the distance E from the counter-weapon B's location coordinates C to the enemy tank's location coordinates A.
7. Calculate the weapon flight time F based on the distance E and speed D.
8. Use Situation_display to show the enemy tank's location coordinates A and obtain the image G.
9. Use address_book to look up General Zhang San's QQ Email H.
10. After F time, use QQ_Email to send the image G to General Zhang San's QQ Email H.

React_data:
Thought: First, I need to use map_search to get the location coordinates A of the enemy tank.
Action: map_search
Action Input: {\"launch\": \"enemy tank\"}
Observation: {'our helicopter': [100, 80], 'enemy helicopter': [170, 45], 'our command post': [0, 2], 'enemy tank': [20, 13], 'our rocket launcher': [100, 120], 'our firing position 1': [50, 70], 'our firing position 2': [150, 170], 'enemy command center': [70, 35], 'enemy anti-tank missile': [50, 100], 'our tank': [32, 21]}
Thought: I've obtained the location coordinates A of the enemy tank. Next, I need to use knowledge_graph to query the counter-weapon B for the enemy tank.
Action: knowledge_graph
Action Input: {\"weapon_query\": \"enemy tank\", \"attribute\": \"counter-weapon\"}
Observation: 'Leopard 2A7 Tank': {'counter-weapon': 'anti-tank missile'}
Thought: I've obtained the counter-weapon B for the enemy tank. Next, I need to use weapon_launch to fire the counter-weapon B at the enemy tank's location coordinates A.
Action: weapon_launch
Action Input: {\"weapon_query\": \"anti-tank missile\", \"target_name\": \"enemy tank\", \"target_coordinate\": \"[20, 13]\"}
Observation: Anti-tank missile launched at enemy tank; target location: [20, 13]
Thought: The weapon has been launched. Next, I need to use map_search to query the location coordinates C of the counter-weapon B.
Action: map_search
Action Input: {\"launch\": \"anti-tank missile\"}
Observation: {'enemy anti-tank missile': [50, 100]}
Thought: I've obtained the location coordinates C of the counter-weapon B. Next, I need to use knowledge_graph to get the speed D of the counter-weapon B.
Action: knowledge_graph
Action Input: {\"weapon_query\": \"anti-tank missile\", \"attribute\": \"speed\"}
Observation: The speed of the anti-tank missile is: 200km/h
Thought: I've obtained the speed D of the counter-weapon B. Next, I need to use distance_calculation to calculate the distance E from the counter-weapon B's location coordinates C to the enemy tank's location coordinates A.
Action: distance_calculation
Action Input: {\"weapon_query\": \"{'anti-tank missile':[50, 100]}\", \"map_dict\": \"{'enemy tank': [20, 13]}\"}
Observation: Here are all the distances from {'anti-tank missile': [50, 100]}: {'enemy tank': '92.0km'}
Thought: I've obtained the distance E from the counter-weapon B's location coordinates C to the enemy tank's location coordinates A. Next, I need to calculate the weapon flight time F based on the distance E and speed D.
Action: python_math
Action Input: {\"math_formulation\": \"92.0/200\"}
Observation: The result is 0.46
Thought: I've calculated the weapon flight time F. Next, I need to use Situation_display to show the enemy tank's location coordinates A and obtain the image G.
Action: Situation_display
Action Input: {\"coordinate\": \"[20, 13]\", \"radius\": \"300\"}
Observation: Displayed the situation map centered at [20, 13] with a radius of 300, the image URL is /ai/ld/picture1.png
Thought: I've obtained the image G. Next, I need to use address_book to look up General Zhang San's QQ Email H.
Action: address_book
Action Input: {\"person_name\": \"Zhang San\"}
Observation: {'email': '45123456@qq.com', 'phone': '13713156111', 'unit': 'Black Fox Tank', 'position': 'Deputy Chief Engineer'}
Thought: I've obtained General Zhang San's QQ Email H. Next, I need to use QQ_Email to send the image G to General Zhang San's QQ Email H.
Action: QQ_Email
Action Input: {\"E-mail_address\": \"45123456@qq.com\", \"E-mail_content\": \"The result of striking the enemy tank as shown in the image: /ai/ld/picture1.png\"}
Observation: Sent the image showing the result of striking the enemy tank: /ai/ld/picture1.png to 45123456@qq.com
Thought: I have completed all tasks.
Final Answer: Through the above steps, I successfully used an anti-tank missile to strike the enemy tank.
```

# Usage Instructions
1. First, modify the config.py configuration file. There are comments already in the [config.py](./config.py) file.
2. Modify the tools_caption.py file, filling in the configurations for the tools that the Agent needs to call (multiple tools can be configured). Below is an example.
```python
tools=[
  {
        "name_for_human": "Knowledge Graph", # name_for_human writes the tool name, which should be self-explanatory,
        "name_for_model": "knowledge_graph", # name_for_model is the function name, which is the actual name called by the tool
        "execute_function": True, # execute_function indicates whether the tool should be actually called to obtain higher quality data when constructing the data.
        "description_for_model": "The Knowledge Graph inputs a type of weapon to obtain its attributes, or inputs an attribute to get all weapons' corresponding attributes", # description_for_model is a prompt for the model, explaining the purpose of the tool. It needs to be clear and concise.
        "example": "Help me check the endurance of the enemy helicopter", # example represents a problem that can be solved using this tool.
        "parameters": [
            {
                "name": "weapon_query", # This is the first parameter name
                "description": "Weapon name", # This is a textual description of the parameter, used to prompt the model. It needs to be self-explanatory.
                "scope": ["Helicopter", 'Leopard 2A7 Tank', 'Black Fox Tank', "Radar", 'Armored Vehicle', 'Sniper Rifle', "Anti-Tank Missile", "Helicopter", "Rocket Launcher", "All Weapons", 'Drone', 'Infantry'],  # The range of values for the parameter
                "required": True, # Whether this parameter is required
                "schema": {"type": "string"}, # The data structure type of this parameter
            },
            {
                "name": "attribute", # This is the second parameter name
                "description": "Weapon attribute", # This is a textual description of the parameter, used to prompt the model. It needs to be self-explanatory.
                "scope": ['Range', 'Carried Weapon', 'Weight', 'Speed', 'Adaptation Scenario', 'Counter-Weapon', 'Endurance', 'Full Load Personnel', 'Flight Altitude', 'Load Capacity', 'All Attributes', 'Detection Range'], # The range of values for the parameter
                "required": True, # Whether this parameter is required
                "schema": {"type": "string"}, # The data structure type of this parameter
            },
        ],
    }
    ]
```
3. For the tools where "execute_function": True in step 2, implementation is needed in the tools_call.py file. Below is an example.
```python
if plugin_name == "knowledge_graph":
        weapon_name = args_dict["weapon_query"] # Parsing of input parameters, must match the name value of the parameter (weapon_query) in step 2
        attribute = args_dict["attribute"] # Parsing of input parameters, must match the name value of the parameter (attribute) in step 2
        kg = {
            "Helicopter": {
                "Flight Altitude": "within 0.3km",
                "Carried Weapon": "Rocket",
                "Counter-Weapon": "Anti-Air Missile",
                "Weight": "3000kg",
                "Speed": "100km/h",
                "Range": "2km",
                "Adaptation Scenario": "Air Combat",
                "Endurance": "500km",
                "Full Load Personnel": "7 people",
                "Load Capacity": "10000kg",
                "Endurance Mileage": "1000km",
            },
            # Additional weapons and attributes omitted here...
            "Sniper Rifle": {"Range": "1.2km", "Weight": "30kg", "Adaptation Scenario": "Assassination"},
        }
        if weapon_name != "All Weapons":
            try:
                return "{}'s {} is:{}".format(
                    weapon_name, attribute, kg[weapon_name][attribute]
                )
            except:
                return kg
        return kg # The return value should be text that is easily understood by the model
```
4. Depending on the specific requirements, modify the main.py file at the bottom, and execute one of the following functions. Make sure that the first step has been configured according to the executed function:
  ```python
  if __name__ == "__main__":
      #get_question() # Obtain simple Agent question data
      #get_complex_question()
      #get_react_data # Obtain Agent execution process data
      get_complex_react_data() # Obtain complex Agent execution process data
  ```
5. After executing `get_react_data()` or `get_complex_react_data()`, the training data saved is in the Qwen training format by LLaMAFactory and can be directly used for SFT (Supervised Fine-Tuning) training.
