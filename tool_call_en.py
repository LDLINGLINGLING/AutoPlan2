tools = [
    {
        "name_for_human": "Image to Text",
        "name_for_model": "image_gen_prompt",
        "excute_function": False,
        "description_for_model": "Service that generates detailed textual descriptions from images. Input an image URL/path to get the description.",
        "example": "Please check the stock price shown in today's image at www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png",
        "parameters": [
            {
                "name": "image_path",
                "description": "Image URL or local path",
                "scope": None,
                "required": True,
                "schema": {"type": "string"},
            }
        ],
    },
    {
        "name_for_human": "Knowledge Graph",
        "name_for_model": "knowledge_graph",
        "excute_function": True,
        "description_for_model": "Query weapon attributes by type or find weapons with specific attributes.",
        "example": "Find the endurance mileage of enemy helicopters",
        "parameters": [
            {
                "name": "weapon_query",
                "description": "Weapon name",
                "scope": ["Helicopter", "Leopard 2A7 Tank", "Black Fox Tank", "Radar", "Armored Vehicle", "Sniper Rifle", "Anti-Tank Missile", "Rocket Launcher", "All Weapons", "UAV", "Infantry"],
                "required": True,
                "schema": {"type": "string"},
            },
            {
                "name": "attribute",
                "description": "Weapon attribute",
                "scope": ["Range", "Armament", "Weight", "Speed", "Scenario", "Countermeasure", "Endurance", "Capacity", "Altitude", "Payload", "All Attributes", "Detection Range"],
                "required": True,
                "schema": {"type": "string"},
            },
        ],
    },
    {
        'name_for_human': 'Google Search',
        'name_for_model': 'google_search',
        'excute_function': True,
        'description_for_model': 'General search engine for internet access, encyclopedia queries, and news updates.',
        "example": "Check the latest economic situation",
        'parameters': [
            {
                'name': 'search_query',
                'description': 'Search keywords/phrases',
                'scope': False,
                'required': True,
                'schema': {'type': 'string'},
            }
        ]
    },
    {
        'name_for_human': 'Military Intel Search',
        'name_for_model': 'military_information_search',
        'excute_function': True,
        'description_for_model': 'Specialized search engine for military networks and intelligence.',
        "example": "Check enemy troop movements",
        'parameters': [
            {
                'name': 'search_query',
                'description': 'Search keywords/phrases',
                'scope': 'All military-related searches',
                'required': True,
                'schema': {'type': 'string'},
            }
        ],
    },
    {
        'name_for_human': 'Address Book',
        'name_for_model': 'address_book',
        'excute_function': True,
        'example': "What is Li Si's phone number?",
        'description_for_model': 'Retrieves personal information like contacts and addresses.',
        'parameters': [
            {
                'name': 'person_name',
                'description': 'Name of person to query',
                'scope': ['Zhang San', 'Li Si', 'Wang Wu', 'Our Command Center', 'Recon Unit', 'Commander'],
                'required': True,
                'schema': {'type': 'string'},
            }
        ],
    },
    {
        'name_for_human': 'Email Client',
        'name_for_model': 'QQ_Email',
        'excute_function': True,
        "example": "Send 'Spy detected, abort meeting' to 403644785@163.com",
        'description_for_model': 'Email tool for sending/receiving messages',
        'parameters': [
            {
                'name': 'E-mail_address',
                'description': 'Recipient email address',
                'scope': None,
                'required': True,
                'schema': {'type': 'string'},
            },
            {'name': "E-mail_content",
             'description': 'Email content',
             'scope': None,
             'required': True,
             'schema': {'type':'string'}
            }
        ],
    },
    {
        'name_for_human': 'Situation Display',
        'name_for_model': 'Situation_display',
        'excute_function': True,
        'example':'Show situational map at enemy HQ [400,200]',
        'description_for_model': 'Displays battlefield situation image based on coordinates and radius',
        'parameters': [
            {
                'name': 'coordinate',
                'description': 'Target coordinates (x,y) as list',
                'scope':'x,y values (0-400 integers)',
                'required': True,
                'schema': {'type': 'list'},
            },
            {
                'name': 'radio',
                'description': 'Display radius in km (default 300km)',
                'scope':'0-560 integer',
                'required': False,
                'schema': {'type': 'string'},
            }
        ],
    },
    {
        'name_for_human': 'Calendar',
        'name_for_model': 'calendar',
        'excute_function': True,
        'example':'What is today\'s date?',
        'description_for_model': 'Tool for getting current date/time',
        'parameters': [
            {
                'name': 'time_query',
                'description':'Target time (e.g. yesterday, today, tomorrow)',
                'required': True,
                'schema': {'type': 'string'},
            }
        ],
    },
    {
        'name_for_human': 'Map Search',
        'name_for_model': 'map_search',
        'excute_function': True,
        'example':'Where is our helicopter now?',
        'description_for_model': 'Queries position information of all map units',
        'parameters': [
            {
                'name': 'lauch',
                'description': 'Unit name or "yes" to activate',
                'scope':['Our Helicopter', 'Enemy Helicopter', 'Our Command Center', 'Enemy Tank', 'Our Rocket Launcher', 'Launch Site 1', 'Launch Site 2', "Enemy Command Center", "Enemy Anti-Tank Missile", 'Our Tank', 'All Units'],
                'required': True,
                'schema': {'type': 'string'},
            }
        ],
    },
    {
        'name_for_human': 'Python Calculator',
        'name_for_model': 'python_math',
        'excute_function': True,
        'example':'100//3+20**2',
        'description_for_model': 'Evaluates mathematical expressions using Python',
        'parameters': [
            {
                'name': 'math_formulation',
                'description': 'Mathematical expression to evaluate',
                'scope':'Basic arithmetic operations',
                'required': True,
                'schema': {'type': 'string'},
            }
        ],
    },
    {
        'name_for_human': 'Weapon Launch',
        'name_for_model': 'weapon_launch',
        'excute_function': True,
        'example':'Launch rocket launcher at enemy position [10,30]',
        'description_for_model': 'Activates weapons to strike targets',
        'parameters': [
            {
                'name': 'weapon_query',
                'description': 'Weapon to launch',
                'scope':"['Our Helicopter','Enemy Helicopter','Our Command Center','Enemy Tank','Our Rocket Launcher','Enemy Anti-Tank Missile','Our Tank'] with 0-400 coordinates",
                'required': True,
                'schema': {'type': 'string'},
            },
            {
                'name': 'target_name',
                'description': "Target unit name",
                'scope': None,
                'required': True,
                'schema': {'type': 'string'},
            },
            {
                'name': 'target_coordinate' ,
                'description': 'Target coordinates [x,y]',
                'scope':"0-400 integers",
                'required': True,
                'schema': {'type': 'string'},
            }
        ],
    },
    {
        'name_for_human': 'Distance Calculator',
        'name_for_model': 'distance_calculation',
        'excute_function': True,
        'example':'Calculate distance between our helicopter[200,300] and command center[100,200]',
        'description_for_model': 'Calculates distances between units on map',
        'parameters': [
            {
                'name': 'weapon_query',
                'description': '{TargetUnit:[x,y]} (single unit)',
                'scope':'Any map unit',
                'required': True,
                'schema': {'type': 'string'},
            },
            {
                'name': 'map_dict',
                'description': '{Unit:[x,y], ...}',
                'scope':'Other units on map',
                'required': True,
                'schema': {'type': 'dict'},
            }
        ],
    }
]
