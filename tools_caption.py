tools = [
    {
        "name_for_human": "图生文",
        "name_for_model": "image_gen_prompt",
        "excute_function": False,  # 是否可以使用这个工具进行函数调用以生成数据
        "description_for_model": "图生文是一个可以看图生成文字描述的服务，输入一张图片的地址，将返回图片详细逼真的表述",
        "example": "帮我看一下www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png这张图片上的今日股价是多少",
        "parameters": [
            {
                "name": "image_path",
                "description": "需要图片描述的URL或者本地地址",
                "scope": None,  # 这个参数的取值范围，如果不限定为None
                "required": True,  # 这个是否必须
                "schema": {"type": "string"},
            }
        ],
    },
    {
        "name_for_human": "知识图谱",
        "name_for_model": "knowledge_graph",
        "excute_function": True,
        "description_for_model": "知识图谱是输入武器种类获取该武器的属性，也可以输入某种属性获得所有武器的该属性",
        "example": "帮我查一下敌方直升机的续航里程",
        "parameters": [
            {
                "name": "weapon_query",
                "description": "武器名称",
                "scope": ["直升机", '豹2A7坦克','黑狐坦克',"雷达",'装甲车','狙击枪', "反坦克导弹", "直升机", "火箭炮", "所有武器",'无人机','步兵'],  # 参数的取值范围
                "required": True,
                "schema": {"type": "string"},
            },
            {
                "name": "attribute",
                "description": "武器的属性",
                "scope": ['射程','携带武器','重量','速度','适应场景','克制武器','续航','满载人数','飞行高度','承载重量','所有属性','探测范围'],
                "required": True,
                "schema": {"type": "string"},
            },
        ],
    },
    {
            'name_for_human': '谷歌搜索',
            'name_for_model': 'google_search',
            'excute_function':True,
            'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
            "example": "帮我查一下最新的经济形势",
            'parameters': [
                {
                    'name': 'search_query',
                    'description': '搜索关键词或短语',
                    'scope':False,
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ]
        },
        {
            'name_for_human': '军事情报搜索',
            'name_for_model': 'military_information_search',
            'excute_function':True,
            'description_for_model': '军事情报搜索是一个通用搜索引擎，可用于访问军事情报网、查询军网、了解军事新闻等。',
            "example": "帮我查一下敌军的走向",
            'parameters': [
                {
                    'name': 'search_query',
                    'description': '搜索关键词或短语',
                    'scope':'所有军事类信息搜索',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '通讯录',
            'name_for_model': 'address_book',
            'excute_function':True,
            'example':'李四的电话号号码是多少?',
            'description_for_model': '通讯录是用来获取个人信息如电话、邮箱地址、公司地址的软件。',
            'parameters': [
                {
                    'name': 'person_name',
                    'description': '被查询者的姓名',
                    'scope':['张三','李四','王五','我方指挥所','侦察部队','指挥官'],
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': 'qq邮箱',
            'name_for_model': 'QQ_Email',
            'excute_function':True,
            "example": "帮我将‘有间谍，停止接头’发到403644785@163.com",
            'description_for_model': 'qq邮箱是一个可以用来发送合接受邮件的工具',
            'parameters': [
                {
                    'name': 'E-mail_address',
                    'description': '对方邮箱的地址 发给对方的内容',
                    'scope':None,
                    'required': True,
                    'schema': {'type': 'string'},
                },
                {'name':"E-mail_content",
                 'description': '发给对方的内容',
                 'scope':None,
                'required': True,
                'schema': {'type':'string'}}
            ],
        },
        {
            'name_for_human': '态势显示',
            'name_for_model': 'Situation_display',
            'excute_function':True,
            'example':'我需要查看敌军总部[400,200]处的态势图，请显示',
            'description_for_model': ':态势显示是通过输入目标位置坐标和显示范围，从而显示当前敌我双方的战场态势图像，并生成图片',
            'parameters': [
                {
                    'name': 'coordinate',
                    'description': '目标位置的x和y坐标，输入一个列表,如[300,200]',
                    'scope':'coordinate_x,coordinate_y的取值都在0到400之间的整数',
                    'required': True,
                    'schema': {'type': 'list'},
                }
                ,
                {
                    'name': 'radio',
                    'description': '态势图像显示的范围半径,单位是km,默认值为300km',
                    'scope':'radio取值都在0到560之间的整数',
                    'required': False,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '万年历',
            'name_for_model': 'calendar',
            'excute_function':True,
            'example':'请告诉我今天的日期',

            'description_for_model': '万年历获取当前时间的工具',
            'parameters': [
                {
                    'name': 'time_query',
                    'description':'目标的时间，例如昨天、今天、明天',
                    'location':'location_query',
                    'scope':None,
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '地图',
            'name_for_model': 'map_search',
            'excute_function':True,
            'example':'我方直升机的地址在哪里',
            'description_for_model': '地图是一个可以查询地图上所有单位位置信息的工具，返回所有敌军的位置信息。',
            'parameters': [
                {
                    'name': 'lauch',
                    'description': '输入单位名称或者yes,yes代表启用地图搜索',
                    'scope':['我方直升机','敌直升机','我方指挥所','敌坦克','我方火箭炮','我方发射阵地1','我方发射阵地2',"敌指挥中心","敌反坦克导弹",'我方坦克','所有单位'],
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': 'python计算器',
            'name_for_model': 'python_math',
            'excute_function':True,
            'example':'100//3+20**2',
            'description_for_model': 'python计算器可以通过python的eval()函数计算出输入的字符串表达式结果并返回,表达式仅包含数字、加减乘除、逻辑运算符',
            'parameters': [
                {
                    'name': 'math_formulation',
                    'description': '根据问题提炼出的python数学表达式,表达式仅包含数字、加减乘除、逻辑运算符',
                    'scope':'加、减、乘、除、平方、开平方等基本算数操作',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '武器发射按钮',
            'name_for_model': 'weapon_launch',
            'excute_function':True,
            'example':'发射火箭炮打击地方火箭炮[10,30]',
            'description_for_model': '武器发射按钮是可以启动指定武器打击指定目标位置工具。',
            'parameters': [
                {
                    'name': 'weapon_query',
                    'description': '启动的打击武器名称',
                    'scope':"weapon_query的取值为['我方直升机','敌直升机','我方指挥所','敌坦克','我方火箭炮','敌反坦克导弹','我方坦克'],x,y坐标都是0到400之间的整数",
                    'required': True,
                    'schema': {'type': 'string'},
                },
                {
                    'name': 'target_name',
                    'description': "被打击的单位名称",
                    'scope':None,
                    'required': True,
                    'schema': {'type': 'string'},
                }
                ,
                {
                    'name': 'target_coordinate' ,
                    'description': '被打击目标的坐标地点比如[x, y]',
                    'scope':"x,y坐标都是0到400之间的整数",
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '数学计算',
            'name_for_model': 'math_model',
            'excute_function':True,
            'example':'根据飞机速度和A、B两地的距离，计算来回一次时间',
            'description_for_model': '使用大语言模型完成一系列的推理问题如基本的加减乘除、最大、最小计算',
            'parameters': [
                {
                    'name': 'question',
                    'description': '当前的问题，需要清楚的给足背景知识',
                    'scope':'加、减、乘、除、平方、开平方等基本算数操作',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '距离计算器',
            'name_for_model': 'distance_calculation',
            'excute_function':True,
            'example':'计算我方直升机[200,300]和其他所有单位的距离',
            'description_for_model': '会将map_dict上所有单位都与weapon_query进行距离计算，返回map_dict上每一个单位和weapon_query的距离',
            'parameters': [
                {
                    'name': 'weapon_query',
                    'description': '{目标单位名称:[目标单位名称的x坐标,目标单位名称的y坐标]},只能是一个单位，比如{我方直升机:[200,300]}',
                    'scope':'weapon_query为地图上任意目标，但是只能是一个单位',
                    'required': True,
                    'schema': {'type': 'string'},
                },
                {
                    'name': 'map_dict',
                    'description': '{被计算的单位名称:[该单位的x坐标,该单位的y坐标],被计算的第二个单位名称:[该单位的x坐标,该单位的y坐标]},比如{地方坦克:[300,150]}',
                    'scope':'map_dict为除目标单位以外的所有地图上单位的名称和位置参数，可以是一个单位也可以是多个单位',
                    'required': True,
                    'schema': {'type': 'dict'},
                }
            ],
        }
]