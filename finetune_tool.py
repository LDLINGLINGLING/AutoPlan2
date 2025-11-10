# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

tools = [
  {
    "name": "knowledge_graph",
    "description": "Query military weapon knowledge graph and information, \n\n      Returns:  \n        str: The attribute value or full knowledge graph  \n    ",
    "parameters": {
      "type": "object",
      "properties": {
        "weapon_query": {
          "title": "Weapon Query",
          "type": "string",
          "description":'Name of the weapon to query. candidate value is one of ["Helicopter","Anti-Tank Missile", "Infantry","Drone","Leopard 2A7 Tank","Black Fox Tank","Black Fox Tank","Rocket Launcher","Radar","Armored Vehicle","Armored Vehicle","Sniper Rifle"]'
        }
      },
      "required": [
        "weapon_query"
      ]
    }
  },
  {
    "name": "QQ_Email",
    "description": "Send email via QQ email system  \n      \n    Returns:  \n        str: Confirmation message  \n    ",
    "parameters": {
      "type": "object",
      "properties": {
        "E-mail_content": {
          "title": "E-mail_content",
          "type": "string",
          "description":"Content of the email to send"
        },
        "E-mail_address": {
          "title": "E-mail_address",
          "type": "string",
          "description":"Recipient email address"
        }
      },
      "required": [
        "E-mail_content",
        "E-mail_address"
      ]
    }
  },
  {
    "name": "calendar",
    "description": "Get current date and time  \n    Returns:  \n        str: Current timestamp in YYYY-MM-DD HH:MM:SS format  \n    ",
    "parameters": {
      "type": "object",
      "properties": 
      {
        "time_query": {
          "title": "time_query",
          "type": "string",
          "description" : "only support Get current date and time"
        }
      },
      "required": ["time_query"]
    }
  },
  {
    "name": "map_search",
    "description": "Get current tactical map with unit positions lauch \n      \n    Returns:  \n        str: Dictionary of unit positions as coordinates  \n    ",
    "parameters": {
      "type": "object",
      "properties": {
        "lauch": {
          "title": "Lauch",
          "type": "string",
          "description":"unit positions lauch.  candidate value is one of ['Our Helicopter', 'Enemy Helicopter', 'Our Command Center', 'Enemy Tank', 'Our Rocket Launcher', 'Launch Site 1', 'Launch Site 2', 'Enemy Command Center', 'Enemy Anti-Tank Missile', 'Our Tank', 'All Units']"
        }
      },
      "required": [
        "lauch"
      ]
    }
  },
  {
    "name": "address_book",
    "description": "Look up contact information for address or personnel.  \n      \n    Returns:  \n        str: Contact information including email, phone, unit, and position  \n    ",
    "parameters": {
      "type": "object",
      "properties": {
        "person_name": {
          "title": "person_name",
          "type": "string",
          "description":"['Li Si','Zhang San','Wang Wu',Our Command Post,'Special Forces','Reconnaissance Unit','Commander']"
        }
      },
      "required": [
        "person_name"
      ]
    }
  },
  {
    "name": "weapon_launch",
    "description": "Launch weapon at specified target  \n      \n    Returns:  \n        str: Launch confirmation message  \n    ",
    "parameters": {
      "type": "object",
      "properties": {
        "weapon_query": {
          "title": "weapon query",
          "type": "string",
          "description":'Type of weapon to launch. candidate value is one of ["Helicopter","Anti-Tank Missile", "Infantry","Drone","Leopard 2A7 Tank","Black Fox Tank","Black Fox Tank","Rocket Launcher","Radar","Armored Vehicle","Armored Vehicle","Sniper Rifle"]'
        },
        "target_name": {
          "title": "target name",
          "type": "string",
          "description":"Name of the target. "
        },
        "target_coordinate": {
          "title": "target coordinate",
          "type": "string",
          "description":"Coordinates of the target. Format: (x, y)"
        }
      },
      "required": [
        "weapon_query",
        "target_name",
        "target_coordinate"
      ]
    }
  },
  {
    "name": "situation_display",
    "description": "Display tactical situation map  \n       Returns:  \n        str: Map display information with image URL  \n    ",
    "parameters": {
      "type": "object",
      "properties": {
        "coordinate": {
          "title": "coordinate",
          "type": "string",
          "description":"Center coordinate for the map. Format: (x, y)"
        },
        "radio": {
          "default": 300,
          "title": "Radio",
          "type": "integer",
          "description":"Radius in km for the map display. Default is 300km"
        }
      },
      "required": [
        "coordinate"
      ]
    }
  },
  {
    "name": "distance_calculation",
    "description": "Calculate distances between units and a specified weapon  \n      \n    Returns:  \n        str: Distance calculations between all units and the specified weapon  \n    ",
    "parameters": {
      "type": "object",
      "properties": {
        "weapon_query": {
          "title": "Weapon Query",
          "type": "string",
          "description":'JSON string of weapon and its coordinates. Format: {"TargetUnit":[x,y]} (single unit). '
        },
        "map_dict": {
          "title": "Map Dict",
          "type": "string",
          "description":"JSON string of all unit positions. Format: {\"UnitName1\":[x1,y1],\"UnitName2\":[x2,y2],...}"
        }
      },
      "required": [
        "weapon_query",
        "map_dict"
      ]
    }
  }
]
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-2B-sft-bf16")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="data/AdvertiseGenChatML/train.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default="data/AdvertiseGenChatML/dev.json",
        metadata={"help": "Path to the test data."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)
    qlora: bool = field(default=False)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.
    Loads the dataset from a json file and preprocesses it.example:
    /data/AdvertiseGenChatML/train.json
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length=4096,
    ):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["label_ids"]:
            if id_ == -100:
                continue
            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    def process_tokens(self,data, tokenizer, tools):
        """
        处理对话数据，生成input_tokens和labels
        
        Args:
            data: 对话数据
            tokenizer: 分词器
            tools: 工具列表
        
        Returns:
            tuple: (input_tokens, labels)
                - input_tokens: 原始tokens列表
                - labels: 将指定区间改为-100的tokens列表
        """
        toolresponse_token1 = 151665
        
        toolresponse_token2 = 151666
        
        # 生成tokens
        tokens = tokenizer.apply_chat_template(
            data,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        # 一次遍历完成所有查找，降低计算复杂度
        indices1 = []
        indices2 = []
        indices3 = None
        
        for i in range(len(tokens)):
            # 查找 toolresponse_token1 和 toolresponse_token2
            if tokens[i] == toolresponse_token1:
                indices1.append(i)
            elif tokens[i] == toolresponse_token2:
                indices2.append(i)
            
            # 查找第一个 [151658,151645,151644] 序列中的 151644
            if indices3 is None and i >= 2:
                if (
                    tokens[i-2] == 151645 and 
                    tokens[i-1] == 198 and
                    tokens[i] == 151644 and 
                    tokens[i-3] != 151658 ):
                    indices3 = i
            
            # # 查找 indices3 后面第一个 [151645,151644] 序列中的 151644
            # if indices4 is None and indices3 is not None and i > indices3 and i >= 1:
            #     if (tokens[i] == 151645 and 
            #         tokens[i+1] == 151644):
            #         indices4 = i
        
        # 创建input_tokens和labels
        input_tokens = tokens.copy()
        labels = tokens.copy()
        
        # 将indices1和indices2对之间的元素改为-100（不包括indices本身）
        assert len(indices1) == len(indices2), "indices1和indices2长度不匹配"
        for idx1, idx2 in zip(indices1, indices2):
            assert idx1 < idx2, f"Error: indices1={idx1} should be < indices2={idx2}"
            # 将idx1+1到idx2-1之间的元素改为-100
            for j in range(idx1 + 1, idx2):
                labels[j] = -100
        
        # 将indices3和indices4之间的元素改为-100（不包括indices3和indices4本身）
        if indices3 is not None :
            
            labels[:indices3] = [-100] * indices3
        
        return input_tokens, labels
    def preprocessing(self, example):

        input_ids, label_ids = self.process_tokens(example, self.tokenizer, tools)
        # 保持input_ids和label_ids长度一致
        input_ids.insert(0, self.tokenizer.eos_token_id)
        label_ids.insert(0, self.ignore_index)  # bos_token不参与损失计算
        
        input_ids.append(self.tokenizer.eos_token_id)
        label_ids.append(self.tokenizer.eos_token_id)  # eos_token参与损失计算
        

        
        # truncate to max len
        input_ids = input_ids[: self.model_max_length]
        label_ids = label_ids[: self.model_max_length]
        attention_mask = [1] * len(input_ids)
        # pad to max len
        input_ids += [self.tokenizer.eos_token_id] * (
            self.model_max_length - len(input_ids)
        )
        label_ids += [self.ignore_index] * (self.model_max_length - len(label_ids))
        attention_mask += [0] * (self.model_max_length - len(attention_mask))
        # convert to pt tensor
        input_ids = torch.LongTensor(input_ids)
        label_ids = torch.LongTensor(label_ids)
        attention_mask = torch.LongTensor(attention_mask)
        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def load_model_and_tokenizer(
    model_path: str,
    max_length: int = 4096,
    use_lora: bool = True,
    qlora: bool = False,
    bf16: bool = False,
    fp16: bool = False,
):
    """load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    if qlora:
        assert use_lora, "use_lora must be True when use_qlora is True"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 是否进行4bit量化
            load_in_8bit=False,  # 是否进行8bit量化
            bnb_4bit_compute_dtype=torch.float16,  # 计算精度设置
            bnb_4bit_quant_storage=torch.uint8,  # 量化权重的储存格式
            bnb_4bit_quant_type="nf4",  # 量化格式，这里用的是正太分布的int4
            bnb_4bit_use_double_quant=True,  # 是否采用双量化，即对zeropoint和scaling参数进行量化
            llm_int8_enable_fp32_cpu_offload=False,  # 是否llm使用int8，cpu上保存的参数使用fp32
            llm_int8_has_fp16_weight=False,  # 是否启用混合精度
            # llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],  # 不进行量化的模块
            llm_int8_threshold=6.0,  # llm.int8()算法中的离群值，根据这个值区分是否进行量化
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            init_lora_weights="gaussian",
            task_type=TaskType.CAUSAL_LM,
            target_modules=(
                ["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj"]
                if model.config.architectures == ["MiniCPM3ForCausalLM"]
                else ["q_proj", "v_proj"]
            ),
            r=64,
            lora_alpha=32,
            lora_dropout=0.1,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        # trainable params: 2,949,120 || all params: 3,010,652,928 || trainable%: 0.09795616002669305
        model.print_trainable_parameters()
        # model.enable_input_require_grads()  # need when using adapter

    return model, tokenizer


if __name__ == "__main__":
    model_path = "/mnt/data/user/tc_agi/yh/models/MiniCPM"
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        max_length=training_args.model_max_length,
        use_lora=training_args.use_lora,
        qlora=training_args.qlora,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
    )

    train_dataset = SupervisedDataset(
        data_path=data_args.train_data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )
    eval_dataset = SupervisedDataset(
        data_path=data_args.eval_data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    # save the incremental PEFT weights, more details can be found in https://huggingface.co/blog/peft
    model.save_pretrained("output_dir")
