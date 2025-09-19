"""
大语言模型(LLM)工具函数模块

该模块提供了与大语言模型相关的工具函数，主要用于：
1. Token计数：精确计算文本的token数量
2. 成本控制：帮助估算LLM API调用成本
3. 文本预处理：为LLM输入做长度和格式检查

核心功能：
- count_tokens: 使用OpenAI的tiktoken库精确计算token数量

技术实现：
- 使用cl100k_base编码器，与GPT-3.5/GPT-4模型兼容
- 支持中英文混合文本的准确计数
- 提供异常处理，确保系统稳定性

使用示例：
    from utils.llm_utils import count_tokens
    
    # 计算文本token数量
    text = "这是一段测试文本，包含中文和English mixed content。"
    token_count = count_tokens(text)
    print(f"Token数量: {token_count}")
    
    # 估算API调用成本（假设每1K tokens $0.002）
    cost = (token_count / 1000) * 0.002
    print(f"预估成本: ${cost:.4f}")
"""

import tiktoken

# 初始化tiktoken编码器，使用cl100k_base编码（与GPT-3.5/GPT-4兼容）
encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    """
    计算文本的token数量
    
    该函数使用OpenAI的tiktoken库精确计算文本的token数量，
    这对于控制LLM API调用成本和确保输入长度限制非常重要。
    
    Args:
        text (str): 要计算token的文本内容
                   支持中英文混合、特殊字符等
                   如果输入为空或非字符串，返回0
        
    Returns:
        int: 文本的token数量
             返回0表示输入无效或计算失败
        
    技术细节：
        - 使用cl100k_base编码器，与GPT-3.5/GPT-4系列模型兼容
        - 编码器会将文本分解为子词单元（subword units）
        - 对于中文文本，通常一个汉字对应1-2个token
        - 对于英文文本，单词可能会被分割为多个token
    """
    # 输入验证：检查文本是否有效
    if not text or not isinstance(text, str):
        return 0
    
    try:
        # 使用tiktoken编码器计算token数量
        # encode()方法将文本转换为token ID列表，len()获取token数量
        return len(encoding.encode(text))
    except Exception as e:
        # 异常处理：确保函数不会因为编码错误而崩溃
        print(f"Token计算错误: {e}")
        return 0
