import json5
import re
import math
import random
def test_function_call(plugin_name, plugin_args):
    args_dict = json5.loads(plugin_args)
    if plugin_name == "knowledge_graph":
        weapon_name = args_dict["weapon_query"]
        attribute = args_dict["attribute"]
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
            "反坦克导弹": {
                "重量": "100kg",
                "射程": "0.5千米",
                "克制武器": "拦截导弹",
                "适应场景": "打击重装甲武器",
                "速度": "200km/h",
            },
            "步兵": {
                "射程": "0.3km",
                "克制武器": "无人机",
                "适应场景": "陆地",
                "速度": "40km/h",
                "重量": "60kg",
                "承载重量": "50kg",
            },
            "无人机": {
                "速度": "100km/h",
                "重量": "10kg",
                "适应场景": "侦察和暗杀",
                "飞行高度": "0.3km以下",
                "克制武器": "电磁攻击",
                "续航": "50km",
            },
            "豹2A7坦克": {
                "速度": "50km/h",
                "携带武器": "激光炮",
                "克制武器": "反坦克导弹",
                "射程": "5km",
                "重量": "10000kg",
                "续航": "1000km",
                "承载重量": "200000kg",
                "满载人数": "5人",
                "适应场景": "野战和掩护步兵",
            },
            "黑狐坦克": {
                "速度": "70km/h",
                "携带武器": "主炮",
                "克制武器": "反坦克导弹",
                "射程": "15km",
                "重量": "10000kg",
                "承载重量": "50000kg",
                "续航": "1000km",
                "满载人数": "5人",
                "适应场景": "野战和掩护步兵",
            },
            "火箭炮": {
                "速度": "4500km/h",
                "重量": "500kg",
                "射程": "1000km",
                "适应场景": "超远程打击",
                "飞行高度": "万米高空",
                "克制武器": "拦截导弹",
            },
            "雷达": {"重量": "5000kg", "探测范围": "2km以上20km以下", "适应场景": "探测敌军"},
            "装甲车": {
                "速度": "80km/h",
                "携带武器": "副炮",
                "克制武器": "穿甲弹",
                "射程": "0.5km",
                "重量": "10000kg",
                "承载重量": "10000kg",
                "续航": "600km",
                "满载人数": "10人",
            },
            "狙击枪": {"射程": "1.2km", "重量": "30kg", "适应场景": "暗杀"},
        }
        if weapon_name != "所有武器":
            try:
                return "{}的{}是:{}".format(
                    weapon_name, attribute, kg[weapon_name][attribute]
                )
            except:
                return kg
        return kg
    elif plugin_name == 'google_search':
        # 使用 SerpAPI 需要在这里填入您的 SERPAPI_API_KEY！
        search_query=json5.loads(plugin_args)['search_query']
        #model.chat(tokenizer,task_split_prompt,history=[])
        # messages = [
        #     {"role": "system", "content":"You are a helpful assistant."},
        #     {"role": "user", "content": '现在请你模拟无所不知的百科全书，直接正面回复以下问题。'+search_query}
        # ]
        # search_query= tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # outputs = llm.generate(
        #     search_query, 
        # )
        # return outputs[0].outputs[0].text
        return 'yes'
        #return query_bing(search_query, max_tries=3,model=model,tokenizer=tokenizer)
    elif plugin_name == 'military_information_search':
        search_query=json5.loads(plugin_args)['search_query']
        # messages = [
        #     {"role": "system", "content":"You are a helpful assistant."},
        #     {"role": "user", "content": '现在请你模拟军事专家，直接正面回复以下问题。'+search_query}
        # ]
        # search_query= tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # outputs = llm.generate(
        #     search_query, 
        # )
        # return outputs[0].outputs[0].text
        return "yes"
        #return query_bing(search_query, max_tries=3,model=model,tokenizer=tokenizer)

    elif plugin_name == 'image_gen':
        import urllib.parse,json

        prompt = json5.loads(plugin_args)["prompt"]#输入的文本
        prompt = urllib.parse.quote(prompt)
        return json.dumps({'image_url': f'https://image.pollinations.ai/{prompt}'.format(prompt=prompt)}, ensure_ascii=False)
    elif plugin_name == 'QQ_Email':
        import urllib.parse,json
        Email_address = json5.loads(plugin_args)["E-mail_address"]
        Email_content = json5.loads(plugin_args)["E-mail_content"]
        return "已将{}发送到{}".format(Email_content,Email_address)
    elif plugin_name=='calendar':
        from datetime import datetime  
        format_str="%Y年%m月%d日 %H:%M:%S"
        # 如果提供了time_str，将字符串解析为datetime对象  
        time = datetime.now()  
        # 将时间格式化为字符串  
        formatted_time = time.strftime(format_str)  
        return formatted_time
    elif plugin_name=='map_search':#这个是地图信息的api
        map_dict={'我方直升机':[100,80],'敌直升机':[170,45],'我方指挥所':[0,2],'敌坦克':[20,13],'我方火箭炮':[100,120],'我方发射阵地1':[50,70],'我方发射阵地2':[150,170],"敌指挥中心": [70, 35],"敌反坦克导弹":[50,100],'我方坦克':[32,21]}
        import json
        args_dict = json.loads(plugin_args)
        #if args_dict['lauch']=='yes':
        return str(map_dict)
    elif plugin_name=='address_book':#这个是地图信息的api
        book_dict={'李四':{'邮箱':'403644786@qq.com','电话':'13077329411','部队':'直升机','职务':'算法工程师'},
                '张三':{'邮箱':'45123456@qq.com','电话':'13713156111','部队':'黑狐坦克','职务':'副总师'},
                '王五':{'邮箱':'45343438@qq.com','电话':'13745432','部队':'指挥所','职务':'C++开发'},
                '我方指挥所':{'邮箱':'15sadf63@qq.com','电话':'062221234'},
                '特种部队':{'邮箱':'112322233@qq.com','电话':'156123459','队长':'赵六'},
                '侦察部队':{'邮箱':'1456412333@qq.com','电话':'056486123135','队长':'周八'},
                '指挥官':{'邮箱':'123456789@qq.com','电话':'6220486123135'}
                }
        import json
        args_dict = json.loads(plugin_args)
        person_name=args_dict['person_name']
        return str(book_dict[person_name])
    elif plugin_name=='weapon_launch':#这里是发射武器的api
        import json
        args_dict = json.loads(plugin_args)
        weapon=args_dict["weapon_query"]
        target_name = args_dict["target_name"]
        coordinate=args_dict["target_coordinate"]
        return '已启动'+weapon+'打击'+str(target_name)+'打击位置：'+str(coordinate)
    elif plugin_name=='Situation_display':#这里是发射武器的api
        import json
        args_dict = json.loads(plugin_args)
        if 'radio' not in args_dict.keys():
            radio=300
        else:
            radio=args_dict["radio"]
        coordinate = args_dict["coordinate"]
        return '已经显示以{}为中心以{}为半径的态势地图,图片地址为{}'.format(coordinate,radio,random.choice(['/ai/ld/picture1.jpg','/ai/liudan/pic.png','/ai/pictures/nature.png']))
    elif plugin_name=='distance_calculation':#这里是计算距离的api
        def distance(query,map_dict):#计算距离
            distance_dict={}
            query_coordinate=list(query.values())[0]
            for weapon,item in map_dict.items():
                if weapon==query:
                    continue
                else:
                    distance_dict[weapon]=str(round(math.sqrt((float(query_coordinate[0])-float(item[0]))**2+(float(query_coordinate[1])-float(item[1]))**2),1))+'km'
            return [query,distance_dict]
        import json,ast
        args_dict = json.loads(plugin_args)
        weapon=json.loads(re.sub(r"'",'"',args_dict["weapon_query"]))#传进来的武器参数
        coordinate=json.loads(re.sub(r"'",'"',args_dict["map_dict"]))#传进来的整个位置信息
        items=list(coordinate.keys())
        distance_list=distance(weapon,coordinate)
        min_distance_unit=min(distance_list[1],key=distance_list[1].get)
        max_distance_unit=max(distance_list[1],key=distance_list[1].get)
        return '以下是所有单位与{}的距离:{}'.format(list(weapon.keys())[0],str(distance_list[1]))
    elif plugin_name=='python_math':#这里是计算距离的api
        import json
        args_dict = json.loads(plugin_args)
        math_formulation=args_dict['math_formulation']
        
        math_formulation=re.sub('km','',math_formulation)
        math_formulation=re.sub('km/h','',math_formulation)
        
        pattern = re.compile(r"[^0-9/+*-/<>=%()](max|min|abs|pow|sqrt|exp)")
        math_formulation=re.sub(pattern,'',math_formulation)
        
        result=eval(math_formulation)

        return "执行结果是{}".format(str(result))
            
    else:
        return '没有找到该工具'