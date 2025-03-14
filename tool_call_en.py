import json5
import re
import math

def function_call(plugin_name, plugin_args, llm, tokenizer):
    args_dict = json5.loads(plugin_args)
    
    if plugin_name == "knowledge_graph":
        weapon_name = args_dict["weapon_query"]
        attribute = args_dict["attribute"]
        kg = {
            "Helicopter": {
                "Flight Altitude": "Within 0.3km",
                "Carried Weapons": "Rockets",
                "Counter Weapons": "Anti-Air Missiles",
                "Weight": "3000kg",
                "Speed": "100km/h",
                "Range": "2km",
                "Applicable Scenarios": "Aerial Combat",
                "Endurance": "500km",
                "Full Capacity": "7 personnel",
                "Load Capacity": "10000kg",
                "Endurance Mileage": "1000km",
            },
            "Anti-Tank Missile": {
                "Weight": "100kg",
                "Range": "0.5km",
                "Counter Weapons": "Intercept Missiles",
                "Applicable Scenarios": "Heavy Armor Strikes",
                "Speed": "200km/h",
            },
            "Infantry": {
                "Range": "0.3km",
                "Counter Weapons": "Drones",
                "Applicable Scenarios": "Ground Combat",
                "Speed": "40km/h",
                "Weight": "60kg",
                "Load Capacity": "50kg",
            },
            "Drone": {
                "Speed": "100km/h",
                "Weight": "10kg",
                "Applicable Scenarios": "Reconnaissance and Assassination",
                "Flight Altitude": "Below 0.3km",
                "Counter Weapons": "Electromagnetic Attacks",
                "Endurance": "50km",
            },
            "Leopard 2A7 Tank": {
                "Speed": "50km/h",
                "Carried Weapons": "Laser Cannon",
                "Counter Weapons": "Anti-Tank Missiles",
                "Range": "5km",
                "Weight": "10000kg",
                "Endurance": "1000km",
                "Load Capacity": "200000kg",
                "Full Capacity": "5 personnel",
                "Applicable Scenarios": "Field Combat and Infantry Support",
            },
            "Black Fox Tank": {
                "Speed": "70km/h",
                "Carried Weapons": "Main Cannon",
                "Counter Weapons": "Anti-Tank Missiles",
                "Range": "15km",
                "Weight": "10000kg",
                "Load Capacity": "50000kg",
                "Endurance": "1000km",
                "Full Capacity": "5 personnel",
                "Applicable Scenarios": "Field Combat and Infantry Support",
            },
            "Rocket Launcher": {
                "Speed": "4500km/h",
                "Weight": "500kg",
                "Range": "1000km",
                "Applicable Scenarios": "Ultra-Long Range Strikes",
                "Flight Altitude": "High Altitude",
                "Counter Weapons": "Intercept Missiles",
            },
            "Radar": {
                "Weight": "5000kg",
                "Detection Range": "2km to 20km",
                "Applicable Scenarios": "Enemy Detection",
            },
            "Armored Vehicle": {
                "Speed": "80km/h",
                "Carried Weapons": "Secondary Cannon",
                "Counter Weapons": "Armor-Piercing Rounds",
                "Range": "0.5km",
                "Weight": "10000kg",
                "Load Capacity": "10000kg",
                "Endurance": "600km",
                "Full Capacity": "10 personnel",
            },
            "Sniper Rifle": {
                "Range": "1.2km",
                "Weight": "30kg",
                "Applicable Scenarios": "Assassination",
            },
        }
        if weapon_name != "All Weapons":
            try:
                return "The {} of {} is: {}".format(attribute, weapon_name, kg[weapon_name][attribute])
            except:
                return kg
        return kg
    
    elif plugin_name == 'google_search':
        search_query = json5.loads(plugin_args)['search_query']
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": 'Simulate an encyclopedia to answer: ' + search_query}
        ]
        search_query = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate(search_query)
        return outputs[0].outputs[0].text
    
    elif plugin_name == 'military_information_search':
        search_query = json5.loads(plugin_args)['search_query']
        messages = [
            {"role": "system", "content": "You are a military expert."},
            {"role": "user", "content": 'Answer military-related: ' + search_query}
        ]
        search_query = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate(search_query)
        return outputs[0].outputs[0].text
    
    elif plugin_name == 'QQ_Email':
        return "The content '{}' has been sent to {}".format(
            json5.loads(plugin_args)["E-mail_content"],
            json5.loads(plugin_args)["E-mail_address"]
        )
    
    elif plugin_name == 'calendar':
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    elif plugin_name == 'map_search':
        map_data = {
            'Our Helicopter': [100,80],
            'Enemy Helicopter': [170,45],
            'Our Command Post': [0,2],
            'Enemy Tank': [20,13],
            'Our Rocket Launcher': [100,120],
            'Our Launch Site 1': [50,70],
            'Our Launch Site 2': [150,170],
            "Enemy Command Center": [70,35],
            "Enemy Anti-Tank Missile": [50,100],
            'Our Tank': [32,21],
        }
        return str(map_data)
    
    elif plugin_name == 'address_book':
        contacts = {
            'Li Si': {'Email': '403644786@qq.com', 'Phone': '13077329411', 'Unit': 'Helicopter', 'Position': 'Algorithm Engineer'},
            'Zhang San': {'Email': '45123456@qq.com', 'Phone': '13713156111', 'Unit': 'Black Fox Tank', 'Position': 'Deputy Chief Engineer'},
            'Wang Wu': {'Email': '45343438@qq.com', 'Phone': '13745432', 'Unit': 'Command Post', 'Position': 'C++ Developer'},
            'Our Command Post': {'Email': '15sadf63@qq.com', 'Phone': '062221234'},
            'Special Forces': {'Email': '112322233@qq.com', 'Phone': '156123459', 'Commander': 'Zhao Liu'},
            'Reconnaissance Unit': {'Email': '1456412333@qq.com', 'Phone': '056486123135', 'Commander': 'Zhou Ba'},
            'Commander': {'Email': '123456789@qq.com', 'Phone': '6220486123135'},
        }
        return str(contacts[json5.loads(plugin_args)['person_name']])
    
    elif plugin_name == 'weapon_launch':
        return 'Activated {} to strike {} at {}'.format(
            args_dict["weapon_query"],
            args_dict["target_name"],
            args_dict["target_coordinate"]
        )
    
    elif plugin_name == 'Situation_display':
        return 'Displaying situation map centered at {} with {}km radius. Image URL: /ai/ld/picture1'.format(
            args_dict["coordinate"],
            args_dict.get("radio", 300)
        )
    
    elif plugin_name == 'distance_calculation':
        def distance(query, map_dict):
            distance_dict = {}
            query_coordinate = list(query.values())[0]
            for weapon, item in map_dict.items():
                if weapon == query:
                    continue
                else:
                    distance_dict[weapon] = str(round(math.sqrt((float(query_coordinate[0])-float(item[0]))**2 + (float(query_coordinate[1])-float(item[1]))**2),1)) + 'km'
            return [query, distance_dict]
        
        weapon = json5.loads(re.sub(r"'", '"', args_dict["weapon_query"]))
        coordinate = json5.loads(re.sub(r"'", '"', args_dict["map_dict"]))
        distance_list = distance(weapon, coordinate)
        return 'Distances between all units and {}: {}'.format(list(weapon.keys())[0], str(distance_list[1]))
    
    elif plugin_name == 'python_math':
        math_formulation = args_dict['math_formulation']
        math_formulation = re.sub('km', '', math_formulation)
        math_formulation = re.sub('km/h', '', math_formulation)
        try:
            result = eval(math_formulation)
            return "The result is {}".format(str(result))
        except:
            pattern = re.compile(r"[^0-9/+*-/<>=%()](max|min|abs|pow|sqrt|exp)")
            math_formulation = re.sub(pattern, '', math_formulation)
            try:
                result = eval(math_formulation)
                return "The result is {}".format(str(result))
            except:
                return 'Execution failed'
    
    else:
        return 'Tool not found'
