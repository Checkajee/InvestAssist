"""
AKShare数据获取工具模块

该模块提供了对AKShare金融数据API的缓存包装，避免重复调用API，提高数据获取效率。
主要功能包括：
1. 数据缓存：基于参数哈希和时间的缓存机制
2. 缓存管理：按小时自动更新缓存，确保数据时效性
3. 文件存储：使用pickle格式高效存储和读取数据
"""

import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
import sys

# 添加项目根目录到Python路径，以便导入配置模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import cfg

import akshare as ak

# 默认缓存目录：utils/akshare_cache/
DEFAULT_AKSHARE_CACHE_DIR = Path(__file__).parent / "akshare_cache"


class CachedAksharePro:
    """
    AKShare数据缓存处理器
    
    该类封装了AKShare API调用，提供智能缓存功能：
    - 基于函数名、参数和时间生成唯一缓存键
    - 按小时自动更新缓存，确保数据时效性
    - 支持自定义缓存目录
    - 提供详细的调试信息
    """
    
    def __init__(self, cache_dir=None):
        """
        初始化缓存处理器
        
        Args:
            cache_dir (str, optional): 自定义缓存目录路径，默认为utils/akshare_cache/
        """
        if not cache_dir:
            self.cache_dir = DEFAULT_AKSHARE_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)
        
        # 确保缓存目录存在
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self, func_name: str, func_kwargs: dict, verbose: bool = False):
        """
        执行AKShare函数并缓存结果（主要接口）
        
        Args:
            func_name (str): AKShare函数名，如"stock_zh_a_hist"
            func_kwargs (dict): 函数参数字典
            verbose (bool): 是否显示详细日志信息
            
        Returns:
            pandas.DataFrame: 函数执行结果
        """
        # 将参数字典转换为JSON字符串，便于后续处理
        func_kwargs_str = json.dumps(func_kwargs)
        return self.run_with_cache(func_name, func_kwargs_str, verbose)

    def run_with_cache(self, func_name: str, func_kwargs: str, verbose: bool = False):
        """
        带缓存的AKShare函数执行核心逻辑
        
        Args:
            func_name (str): AKShare函数名
            func_kwargs (str): JSON格式的函数参数字符串
            verbose (bool): 是否显示详细日志信息
            
        Returns:
            pandas.DataFrame: 函数执行结果
            
        缓存策略：
        1. 基于参数生成MD5哈希值作为缓存标识
        2. 添加小时级时间戳，确保每小时更新一次缓存
        3. 按函数名分目录存储缓存文件
        """
        # 解析参数字符串
        func_kwargs = json.loads(func_kwargs)
        
        # 生成参数哈希值，确保相同参数使用相同缓存
        args_hash = hashlib.md5(str(func_kwargs).encode()).hexdigest()
        
        # 添加小时级时间戳，实现按小时更新缓存
        trigger_time = datetime.now().strftime("%Y%m%d%H")
        args_hash = f"{args_hash}_{trigger_time}"
        
        # 按函数名创建子目录
        func_cache_dir = self.cache_dir / func_name
        if not func_cache_dir.exists():
            func_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建缓存文件路径
        func_cache_file = func_cache_dir / f"{args_hash}.pkl"
        
        # 检查缓存是否存在
        if func_cache_file.exists():
            if verbose:
                print(f"从缓存加载结果: {func_cache_file}")
            with open(func_cache_file, "rb") as f:
                return pickle.load(f)
        else:
            # 缓存未命中，调用AKShare API
            if verbose:
                print(f"缓存未命中，调用API: {func_name}, 参数: {func_kwargs}")
            
            # 动态调用AKShare函数
            result = getattr(ak, func_name)(**func_kwargs)
            
            if verbose:
                print(f"保存结果到缓存: {func_cache_file}")
            
            # 保存结果到缓存文件
            with open(func_cache_file, "wb") as f:
                pickle.dump(result, f)
            return result


# 创建全局缓存实例，供其他模块直接使用
akshare_cached = CachedAksharePro()

if __name__ == "__main__":
    stock_sse_summary_df = akshare_cached.run(
        func_name="stock_sse_summary", 
        func_kwargs={},
        verbose=True
    )
    print(stock_sse_summary_df)