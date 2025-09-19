"""
OpenAI模型实现

此模块为OpenAI模型提供BaseAgentModel的实现。
采用异步优先的方法，主要实现是异步流式方法(a_stream_run)。

主要功能：
- 支持OpenAI API的同步和异步调用
- 实现流式和非流式响应处理
- 提供重试机制和错误处理
- 支持推理模型(reasoner)的特殊处理
- 自动配置管理和客户端初始化
"""

import os
import sys
import httpx
import openai
import asyncio
from pathlib import Path
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from typing import Dict, List, Optional, AsyncIterator, Callable
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import cfg
from .base_agent_model import (
    BaseAgentModel,
    AsyncResponseStream,
    StreamingChunk,
    ModelResponse
)

class LLMModelConfig:
    """
    LLM模型配置类
    
    用于存储和管理LLM模型的配置参数，包括API密钥、基础URL、
    重试设置、超时设置等。
    """
    
    def __init__(self, model_name: str, api_key: str, base_url: str,
                 max_retries: int = 3, retry_delay: float = 20.0, timeout: float = 60.0, extra_headers: dict = None, proxys: dict = None):
        """
        初始化LLM模型配置
        
        参数:
            model_name (str): 模型名称，如"deepseek-chat"
            api_key (str): API密钥
            base_url (str): API基础URL
            max_retries (int, 可选): 最大重试次数，默认为3
            retry_delay (float, 可选): 重试延迟时间(秒)，默认为20.0
            timeout (float, 可选): 请求超时时间(秒)，默认为60.0
            extra_headers (dict, 可选): 额外的HTTP头，默认为None
            proxys (dict, 可选): 代理设置，默认为None
        """
        self.model_name = model_name    # 模型名称
        self.api_key = api_key          # API密钥
        self.base_url = base_url        # API基础URL
        self.max_retries = max_retries  # 最大重试次数
        self.retry_delay = retry_delay  # 重试延迟时间
        self.timeout = timeout          # 请求超时时间
        self.extra_headers = extra_headers  # 额外HTTP头
        self.proxys = proxys            # 代理设置


class LLMModel(BaseAgentModel):
    """
    OpenAI模型实现类
    
    此类提供BaseAgentModel的具体实现。
    遵循异步优先的方法，主要实现a_stream_run方法，
    而所有其他方法(run, a_run, stream_run)由基类处理。
    
    主要特性：
    - 支持同步和异步OpenAI客户端
    - 自动重试机制和错误处理
    - 支持代理和自定义HTTP头
    - 处理推理模型的特殊参数
    """
    
    def __init__(
        self,
        config: LLMModelConfig,
        **kwargs
    ):
        """
        初始化OpenAI模型
        
        参数:
            config (LLMModelConfig): LLM模型配置对象
            **kwargs: 额外的配置参数
        """
        super().__init__(config, **kwargs)
        
        # 保存配置信息
        self.config = config
        self.model_name = config.model_name
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.extra_headers = config.extra_headers
        self.proxys = config.proxys
        
        # 如果API密钥或基础URL未提供，尝试从环境变量获取
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.base_url is None:
            self.base_url = os.environ.get("OPENAI_BASE_URL")

        # 初始化同步客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.Client(proxy=self.proxys) if self.proxys else None
        )
        
        # 初始化异步客户端
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.AsyncClient(proxy=self.proxys) if self.proxys else None
        )

        # 如果提供了额外HTTP头，应用到客户端
        if self.extra_headers is not None:
            self.client = self.client.with_options(
                default_headers=self.extra_headers)
            self.async_client = self.async_client.with_options(
                default_headers=self.extra_headers)
    
    def _process_chunk(self, chunk: ChatCompletionChunk) -> StreamingChunk[str]:
        """
        处理来自OpenAI的流式数据块
        
        从OpenAI的ChatCompletionChunk中提取内容，并转换为标准的StreamingChunk格式。
        特别处理推理模型(reasoner)的reasoning_content字段。
        
        参数:
            chunk (ChatCompletionChunk): OpenAI返回的数据块
            
        返回:
            StreamingChunk[str]: 标准化的流式数据块对象
        """
        # 从数据块中提取内容
        # 检查是否有推理内容（推理模型特有）
        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
            content = chunk.choices[0].delta.reasoning_content  # 推理过程内容
            is_reasoning = True
        else:
            content = chunk.choices[0].delta.content or ""  # 普通内容
            is_reasoning = False
            
        # 检查是否为最后一个数据块
        is_finished = len(chunk.choices) > 0 and chunk.choices[0].finish_reason is not None
        
        return StreamingChunk(
            content=content,          # 数据块内容
            is_finished=is_finished,  # 是否结束标志
            raw_chunk=chunk,         # 原始数据块
            is_reasoning=is_reasoning # 是否为推理内容
        )

    async def a_run_with_semaphore(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        timeout: Optional[float] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        **kwargs
    ) -> ModelResponse[str]:
        """
        使用信号量异步运行模型并流式返回响应
        
        这是所有其他方法(run, a_run, stream_run)将使用的基础实现。
        使用信号量来控制并发请求数量，避免过多并发请求导致API限制。
        
        参数:
            messages (List[Dict[str, str]]): 消息字典列表
            temperature (float, 可选): 采样温度，默认为0.7
            max_tokens (Optional[int], 可选): 最大token数量
            max_retries (Optional[int], 可选): 最大重试次数
            retry_delay (Optional[float], 可选): 重试延迟时间
            timeout (Optional[float], 可选): 超时时间
            semaphore (Optional[asyncio.Semaphore], 可选): 并发控制信号量
            **kwargs: 额外的模型特定参数
            
        返回:
            ModelResponse[str]: 模型响应对象，失败时返回None
        """
        # 使用信号量控制并发
        async with semaphore:
            try:
                response = await self.a_run(
                    messages, 
                    temperature=temperature, 
                    max_tokens=max_tokens, 
                    max_retries=max_retries, 
                    retry_delay=retry_delay, 
                    timeout=timeout, 
                    **kwargs
                )
                return response
            except Exception as e:
                print(f"Error: {e}")
                return None


    async def a_run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        verbose: bool = False,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        timeout: Optional[float] = None,
        post_process_func: Optional[Callable[[str], str]] = None,
        **kwargs
    ) -> ModelResponse[str]:
        """
        异步运行模型并返回完整响应
        
        这是a_stream_run()的包装器，收集所有数据块并组合成单个响应。
        子类应该实现a_stream_run()作为主要方法，此方法将自动处理。
        
        参数:
            messages (List[Dict[str, str]]): 消息字典列表，包含'role'和'content'键
            temperature (float, 可选): 采样温度(0.0到1.0)，默认为0.7
            max_tokens (Optional[int], 可选): 生成的最大token数量，默认为None
            verbose (bool, 可选): 是否打印详细输出，默认为False
            max_retries (Optional[int], 可选): 最大重试次数，默认为None
            retry_delay (Optional[float], 可选): 重试延迟时间，默认为None
            timeout (Optional[float], 可选): 超时时间，默认为None
            post_process_func (Optional[Callable[[str], str]], 可选): 后处理函数，默认为None
            **kwargs: 额外的模型特定参数
            
        返回:
            ModelResponse[str]: 包含生成内容的模型响应对象
        """

        # 设置默认值
        if max_retries is None:
            max_retries = getattr(self, 'config', LLMModelConfig("", "", "")).max_retries
        if retry_delay is None:
            retry_delay = getattr(self, 'config', LLMModelConfig("", "", "")).retry_delay
        if timeout is None:
            timeout = 60

        # 重试机制
        for attempt in range(max_retries + 1):
            try:
                # 获取流式响应
                stream = await self.a_stream_run(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    **kwargs
                )
                
                # 收集所有数据块
                reasoning_content = ""  # 推理过程内容
                full_content = ""      # 主要内容
                raw_chunks = []        # 原始数据块列表
                
                # 异步迭代流式响应
                async for chunk in stream:
                    if chunk.is_reasoning:
                        reasoning_content += chunk.content  # 添加推理内容
                    else:
                        full_content += chunk.content      # 添加主要内容
                    
                    # 保存原始数据块
                    if chunk.raw_chunk is not None:
                        raw_chunks.append(chunk.raw_chunk)
                        
                        # 如果启用详细模式，实时打印内容
                        if verbose:
                            print(chunk.content, end="", flush=True)
            
                # 应用后处理函数（如果提供）
                if post_process_func is not None:
                    proc_response = post_process_func(full_content)
                else:
                    proc_response = None

                # 创建包含收集内容的响应对象
                return ModelResponse(
                    content=self.postprocess_response(full_content),  # 后处理主要内容
                    reasoning_content=reasoning_content,              # 推理内容
                    model_name=self.model_name,                       # 模型名称
                    raw_response=raw_chunks if raw_chunks else None,  # 原始响应数据
                    proc_response=proc_response                      # 后处理响应
                )
            except Exception as e:
                if attempt < max_retries:
                    print(f"🔄 LLM API调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {e}")
                    print(f"⏳ 等待 {retry_delay} 秒后重试...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    print(f"❌ LLM API调用最终失败，已重试 {max_retries} 次: {type(e).__name__}: {e}")
                    raise
    

    async def a_stream_run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> AsyncResponseStream[str]:
        """
        异步运行模型并流式返回响应（带重试机制）
        
        这是所有其他方法(run, a_run, stream_run)将使用的主要实现。
        提供自动重试机制和超时处理，确保API调用的可靠性。
        
        参数:
            messages (List[Dict[str, str]]): 消息字典列表，包含'role'和'content'键
            temperature (float, 可选): 采样温度(0.0到1.0)，默认为0.7
            max_tokens (Optional[int], 可选): 生成的最大token数量，默认为None
            max_retries (Optional[int], 可选): 最大重试次数，默认为None（使用配置默认值3）
            retry_delay (Optional[float], 可选): 重试延迟时间(秒)，默认为None（使用配置默认值20.0）
            timeout (Optional[float], 可选): 每次尝试的超时时间(秒)，默认为None（使用配置默认值60.0）
            **kwargs: 额外的模型特定参数
            
        返回:
            AsyncResponseStream[str]: 产生生成内容数据块的异步响应流
            
        异常:
            - 重试次数用完后抛出最后一次的异常
            - 支持超时、连接错误等常见API错误的自动重试
        """
        
        # 使用配置中的默认值（如果未指定）
        if max_retries is None:
            max_retries = getattr(self, 'config', LLMModelConfig("", "", "")).max_retries
        if retry_delay is None:
            retry_delay = getattr(self, 'config', LLMModelConfig("", "", "")).retry_delay
        if timeout is None:
            timeout = getattr(self, 'config', LLMModelConfig("", "", "")).timeout
        
        # 重试循环
        for attempt in range(max_retries + 1):
            try:
                # 使用asyncio.wait_for设置超时
                return await asyncio.wait_for(
                    self._internal_a_stream_run(messages, temperature, max_tokens, **kwargs),
                    timeout=timeout
                )
            except (
                asyncio.TimeoutError,        # asyncio超时
                openai.APITimeoutError,      # OpenAI API超时
                openai.APIConnectionError,   # OpenAI API连接错误
                ConnectionError,             # 通用连接错误
                TimeoutError                 # 通用超时错误
            ) as e:
                if attempt < max_retries:
                    print(f"🔄 LLM API调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {e}")
                    print(f"⏳ 等待 {retry_delay} 秒后重试...")
                    await asyncio.sleep(retry_delay)  # 等待后重试
                    continue
                else:
                    print(f"❌ LLM API调用最终失败，已重试 {max_retries} 次: {type(e).__name__}: {e}")
                    raise  # 重试次数用完，抛出异常


    async def _internal_a_stream_run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncResponseStream[str]:
        """
        内部异步流式运行实现（无重试逻辑）
        
        这是实际的API调用实现，不包含重试逻辑。
        负责处理消息预处理、参数准备、API调用和响应流处理。
        
        参数:
            messages (List[Dict[str, str]]): 消息字典列表，包含'role'和'content'键
            temperature (float, 可选): 采样温度(0.0到1.0)，默认为0.7
            max_tokens (Optional[int], 可选): 生成的最大token数量，默认为None
            **kwargs: 额外的模型特定参数
            
        返回:
            AsyncResponseStream[str]: 产生生成内容数据块的异步响应流
        """
        # 预处理消息
        processed_messages = self.preprocess_messages(messages)
        
        # 准备API调用参数
        params = {
            "model": self.model_name,        # 模型名称
            "messages": processed_messages,  # 处理后的消息
            "temperature": temperature,      # 采样温度
            "stream": True,                 # 启用流式响应
            **kwargs                        # 其他参数
        }
        
        # 如果指定了最大token数量，添加到参数中
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # 处理推理模型特殊参数
        if 'thinking' in params:
            thinking_flag = params.pop('thinking')  # 移除thinking参数
            if thinking_flag:
                params['extra_body'] = {"thinking": {"type": "enabled"}}   # 启用推理
            else:
                params['extra_body'] = {"thinking": {"type": "disabled"}}  # 禁用推理

        # 调用OpenAI API
        stream = await self.async_client.chat.completions.create(**params)
        
        # 创建异步迭代器来处理数据块
        async def chunk_iterator() -> AsyncIterator[StreamingChunk[str]]:
            """异步数据块迭代器"""
            async for chunk in stream:
                if not chunk.choices:  # 跳过空的数据块
                    continue
                yield self._process_chunk(chunk)  # 处理并产生数据块
        
        return AsyncResponseStream(
            iterator=chunk_iterator(),
            model_name=self.model_name
        )

# 全局LLM配置和实例
# =======================

# 创建普通模式的全局LLM配置
GLOBAL_LLM_CONFIG = LLMModelConfig(
    model_name=cfg.llm["model_name"],    # 从配置文件读取模型名称
    api_key=cfg.llm["api_key"],          # 从配置文件读取API密钥
    base_url=cfg.llm["base_url"]         # 从配置文件读取基础URL
)

# 创建普通模式的全局LLM实例
GLOBAL_LLM = LLMModel(GLOBAL_LLM_CONFIG)

# 尝试创建思考模式的全局LLM配置和实例
try:
    GLOBAL_THINKING_LLM_CONFIG = LLMModelConfig(
        model_name=cfg.llm_thinking["model_name"],    # 从配置文件读取思考模型名称
        api_key=cfg.llm_thinking["api_key"],          # 从配置文件读取思考模型API密钥
        base_url=cfg.llm_thinking["base_url"]         # 从配置文件读取思考模型基础URL
    )
    GLOBAL_THINKING_LLM = LLMModel(GLOBAL_THINKING_LLM_CONFIG)
except Exception as e:
    print(f"加载thinking模型失败，使用llm模型替代: {e}")
    # 如果思考模型加载失败，使用普通模型作为替代
    GLOBAL_THINKING_LLM = GLOBAL_LLM

# VLM模型已移除，vision能力不可用
GLOBAL_VISION_LLM = None

# 模块测试入口
if __name__ == "__main__":
    pass
