from abc import ABC, abstractmethod
from typing import Any, Dict, List
from openai import AzureOpenAI
from vllm import LLM

class BaseLLM(ABC):
    def __init__(self, **kwargs: Any) -> None:
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response based on the given prompt."""
        pass


class LocalLLM(BaseLLM):
    def __init__(self, model: str, tensor_parallel_size: int, max_model_len: int, dtype: str, 
                 trust_remote_code: bool, enforce_eager: bool, gpu_memory_utilization: float) -> None:
        super().__init__()
        # 假设有一个外部库提供了一个名为LLM的类，这里我们只是模拟初始化
        self.model = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    
    def generate(self, prompt: str, **kwargs: Any) -> str:
        # 这里假设generate方法接收一个prompt并返回生成的文本
        # 以及可能的一些额外参数
        return self.model.generate(prompt, **kwargs)


class APILLM(BaseLLM):
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str, model: str) -> None:
        super().__init__()
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.model = model
        self.client = AzureOpenAI(api_key=self.api_key, azure_endpoint=self.azure_endpoint, api_version=self.api_version)
    
    def generate(self, prompt: str, **kwargs: Any) -> str:
        # 创建消息列表
        messages = [{"role": "user", "content": prompt}]
        
        # 调用API并获取响应
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            **kwargs
        )
        
        # 提取并返回生成的文本
        return response.choices[0].message.content