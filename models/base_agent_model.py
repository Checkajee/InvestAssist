"""
基础智能体模型模块

此模块定义了智能体模型的抽象基类。
所有模型实现都应该继承自这个类。

主要功能：
- 定义模型响应的数据结构
- 提供同步和异步的模型调用接口
- 支持流式和非流式响应
- 实现重试机制和错误处理
"""

from abc import ABC, abstractmethod
import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Iterator, TypeVar, Generic

# 泛型类型变量，用于响应内容类型
T = TypeVar('T')


class ModelResponse(Generic[T]):
    """
    模型响应基类
    
    用于封装模型返回的响应数据，包括内容、推理过程、模型名称等信息。
    支持泛型，可以处理不同类型的响应内容。
    """
    
    def __init__(self, content: T, reasoning_content: T, model_name: str, raw_response: Any = None, proc_response: Any = None):
        """
        初始化模型响应对象
        
        参数:
            content (T): 模型生成的主要内容
            reasoning_content (T): 模型的推理过程内容（如果有的话）
            model_name (str): 生成响应的模型名称
            raw_response (Any, 可选): 原始响应数据，用于调试和分析
            proc_response (Any, 可选): 处理后的响应数据
        """
        self.content = content                    # 主要响应内容
        self.reasoning_content = reasoning_content # 推理过程内容
        self.model_name = model_name             # 模型名称
        self.raw_response = raw_response         # 原始响应数据
        self.proc_response = proc_response       # 处理后响应数据



class StreamingChunk(Generic[T]):
    """
    流式响应数据块类
    
    表示流式响应中的一个数据块，包含内容、是否结束、原始数据等信息。
    支持泛型，可以处理不同类型的数据块内容。
    """
    
    def __init__(self, content: T, is_finished: bool = False, raw_chunk: Any = None, is_reasoning: bool = False):
        """
        初始化流式数据块对象
        
        参数:
            content (T): 数据块的内容
            is_finished (bool, 可选): 是否为最后一个数据块，默认为False
            raw_chunk (Any, 可选): 原始数据块，用于调试和分析
            is_reasoning (bool, 可选): 是否为推理过程内容，默认为False
        """
        self.content = content        # 数据块内容
        self.is_finished = is_finished # 是否结束标志
        self.raw_chunk = raw_chunk   # 原始数据块
        self.is_reasoning = is_reasoning # 是否为推理内容


class ResponseStream(Generic[T]):
    """
    同步流式响应类
    
    用于处理同步的流式响应，提供迭代器接口来逐个获取响应数据块。
    支持泛型，可以处理不同类型的响应内容。
    """
    
    def __init__(self, iterator: Iterator[StreamingChunk[T]], model_name: str):
        """
        初始化同步响应流对象
        
        参数:
            iterator (Iterator[StreamingChunk[T]]): 产生StreamingChunk对象的迭代器
            model_name (str): 生成流的模型名称
        """
        self._iterator = iterator  # 内部迭代器
        self.model_name = model_name  # 模型名称
    
    def __iter__(self) -> Iterator[StreamingChunk[T]]:
        """
        返回迭代器
        
        返回:
            Iterator[StreamingChunk[T]]: 用于迭代响应数据块的迭代器
        """
        return self._iterator


class AsyncResponseStream(Generic[T]):
    """
    异步流式响应类
    
    用于处理异步的流式响应，提供异步迭代器接口来逐个获取响应数据块。
    支持泛型，可以处理不同类型的响应内容。
    """
    
    def __init__(self, iterator: AsyncIterator[StreamingChunk[T]], model_name: str):
        """
        初始化异步响应流对象
        
        参数:
            iterator (AsyncIterator[StreamingChunk[T]]): 产生StreamingChunk对象的异步迭代器
            model_name (str): 生成流的模型名称
        """
        self._iterator = iterator  # 内部异步迭代器
        self.model_name = model_name  # 模型名称
    
    def __aiter__(self) -> AsyncIterator[StreamingChunk[T]]:
        """
        返回异步迭代器
        
        返回:
            AsyncIterator[StreamingChunk[T]]: 用于异步迭代响应数据块的迭代器
        """
        return self._iterator


class BaseAgentModel(ABC):
    """
    代理模型抽象基类
    
    此类定义了所有模型实现必须遵循的接口。
    采用异步优先的设计，主要实现是异步流式方法(a_stream_run)。
    所有其他方法(run, stream_run, a_run)都是围绕这个主要方法的包装器。
    
    实现子类时，首先专注于实现a_stream_run，
    其他方法将自动处理。
    
    主要特性：
    - 支持同步和异步调用
    - 支持流式和非流式响应
    - 提供消息预处理和后处理接口
    - 实现重试机制和错误处理
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        初始化模型
        
        参数:
            model_name (str): 模型名称
            **kwargs: 额外的模型特定配置参数
        """
        self.model_name = model_name  # 模型名称
        self.config = kwargs         # 配置参数
    
    def run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse[str]:
        """
        同步运行模型并返回完整响应
        
        这是a_run()的同步包装器。子类应该实现a_run()作为主要方法，
        所有其他方法将自动处理。
        
        参数:
            messages (List[Dict[str, str]]): 消息字典列表，包含'role'和'content'键
            temperature (float, 可选): 采样温度(0.0到1.0)，控制输出的随机性，默认为0.7
            max_tokens (Optional[int], 可选): 生成的最大token数量，默认为None
            **kwargs: 额外的模型特定参数
            
        返回:
            ModelResponse[str]: 包含生成内容的模型响应对象
        """
        # 使用asyncio.run在同步环境中运行异步方法
        return asyncio.run(self.a_run(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ))

    async def a_run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ) -> ModelResponse[str]:
        """
        异步运行模型并返回完整响应
        
        这是a_stream_run()的包装器，收集所有数据块并组合成单个响应。
        子类应该实现a_stream_run()作为主要方法，此方法将自动处理。
        
        参数:
            messages (List[Dict[str, str]]): 消息字典列表，包含'role'和'content'键
            temperature (float, 可选): 采样温度(0.0到1.0)，控制输出的随机性，默认为0.7
            max_tokens (Optional[int], 可选): 生成的最大token数量，默认为None
            verbose (bool, 可选): 是否打印详细输出，默认为False
            **kwargs: 额外的模型特定参数
            
        返回:
            ModelResponse[str]: 包含生成内容的模型响应对象
        """
        # 获取流式响应
        stream = await self.a_stream_run(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # 收集所有数据块
        reasoning_content = ""  # 推理过程内容
        full_content = ""       # 主要内容
        raw_chunks = []         # 原始数据块列表
        
        # 异步迭代流式响应
        async for chunk in stream:
            if chunk.is_reasoning:
                # 如果是推理内容，添加到推理内容中
                reasoning_content += chunk.content
            else:
                # 否则添加到主要内容中
                full_content += chunk.content
            
            # 保存原始数据块用于调试
            if chunk.raw_chunk is not None:
                raw_chunks.append(chunk.raw_chunk)
                
                # 如果启用详细模式，实时打印内容
                if verbose:
                    print(chunk.content, end="", flush=True)
        
        # 创建包含收集内容的响应对象
        return ModelResponse(
            content=self.postprocess_response(full_content),  # 后处理主要内容
            reasoning_content=reasoning_content,              # 推理内容
            model_name=self.model_name,                       # 模型名称
            raw_response=raw_chunks if raw_chunks else None   # 原始响应数据
        )
    
    def stream_run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ResponseStream[str]:
        """
        同步运行模型并流式返回响应
        
        这是a_stream_run()的同步包装器。子类应该实现a_stream_run()作为主要方法，
        此方法将自动处理。
        
        参数:
            messages (List[Dict[str, str]]): 消息字典列表，包含'role'和'content'键
            temperature (float, 可选): 采样温度(0.0到1.0)，控制输出的随机性，默认为0.7
            max_tokens (Optional[int], 可选): 生成的最大token数量，默认为None
            **kwargs: 额外的模型特定参数
            
        返回:
            ResponseStream[str]: 产生生成内容数据块的响应流
        """
        # 对于同步流式处理，我们需要在新的事件循环中运行异步方法
        # 并将所有数据块收集到列表中，然后逐个产生
        
        # 运行异步方法并收集所有数据块
        async def collect_chunks():
            """异步收集所有数据块的内部函数"""
            chunks = []  # 存储所有数据块
            async_stream = await self.a_stream_run(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            # 异步迭代流式响应并收集数据块
            async for chunk in async_stream:
                chunks.append(chunk)
            return chunks
            
        # 运行异步函数并获取所有数据块
        loop = asyncio.new_event_loop()  # 创建新的事件循环
        try:
            chunks = loop.run_until_complete(collect_chunks())  # 运行异步函数
        finally:
            loop.close()  # 确保关闭事件循环
            
        # 创建同步迭代器，逐个产生收集的数据块
        def sync_iterator():
            """同步迭代器函数"""
            for chunk in chunks:
                yield chunk
                
        return ResponseStream(
            iterator=sync_iterator(),
            model_name=self.model_name
        )
    
    @abstractmethod
    async def a_stream_run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncResponseStream[str]:
        """
        异步运行模型并流式返回响应（抽象方法）
        
        这是子类必须实现的主要实现方法。
        所有其他方法(run, a_run, stream_run)都是围绕此方法的包装器。
        
        参数:
            messages (List[Dict[str, str]]): 消息字典列表，包含'role'和'content'键
            temperature (float, 可选): 采样温度(0.0到1.0)，控制输出的随机性，默认为0.7
            max_tokens (Optional[int], 可选): 生成的最大token数量，默认为None
            **kwargs: 额外的模型特定参数
            
        返回:
            AsyncResponseStream[str]: 产生生成内容数据块的异步响应流
            
        注意:
            这是一个抽象方法，子类必须实现此方法来提供具体的模型调用逻辑。
        """
        pass
    
    def preprocess_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        在发送到模型之前预处理消息
        
        子类可以重写此方法来实现自定义预处理逻辑。
        例如：添加系统提示、格式化消息、过滤敏感内容等。
        
        参数:
            messages (List[Dict[str, str]]): 消息字典列表
            
        返回:
            List[Dict[str, str]]: 预处理后的消息字典列表
            
        注意:
            默认实现直接返回原始消息，子类可以根据需要重写此方法。
        """
        return messages
    
    def postprocess_response(self, response: str) -> str:
        """
        后处理模型的响应
        
        子类可以重写此方法来实现自定义后处理逻辑。
        例如：格式化输出、添加标记、清理内容等。
        
        参数:
            response (str): 模型的原始响应
            
        返回:
            str: 后处理后的响应
            
        注意:
            默认实现直接返回原始响应，子类可以根据需要重写此方法。
        """
        return response
