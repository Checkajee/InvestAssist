"""
日期时间处理工具模块

该模块提供了金融交易相关的日期时间处理功能，主要用于：
1. 智能交易日计算：根据交易时间规则自动判断交易日
2. 交易日历处理：处理周末和节假日对交易日的影响
3. 时间格式化：提供多种日期格式输出

核心功能：
- get_current_datetime: 获取当前时间或指定时间
- get_previous_trading_date: 计算上一个交易日
- get_smart_trading_date: 智能交易日判断（15:30分界点）
- get_report_date: 生成中文格式的报告日期
"""

from datetime import datetime, timedelta


def get_current_datetime(trigger_time: str) -> str:
    """
    获取当前时间或返回指定的触发时间
    
    该函数用于统一时间处理，如果提供了trigger_time则使用指定时间，
    否则返回当前系统时间。
    
    Args:
        trigger_time (str): 指定的触发时间，格式：YYYY-MM-DD HH:MM:SS
        
    Returns:
        str: 格式化后的时间字符串，格式：YYYY-MM-DD HH:MM:SS
        
    使用场景：
        - 系统定时任务中获取执行时间
        - 手动触发分析时使用指定时间
    """
    if trigger_time:
        return trigger_time
    else:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_previous_trading_date(trigger_time: str, output_format: str = "%Y%m%d") -> str:
    """
    获取指定时间的上一个交易日
    
    该函数实现了基础的交易日计算逻辑：
    1. 自动跳过周末（周六、周日）
    2. 简化实现，未考虑法定节假日
    
    Args:
        trigger_time (str): 触发时间，格式：YYYY-MM-DD HH:MM:SS
        output_format (str): 输出日期格式，默认：%Y%m%d
    
    Returns:
        str: 上一个交易日，格式：YYYYMMDD
        
    交易日计算规则：
        - 如果前一天是周日，回退到周五
        - 如果前一天是周六，回退到周五
        - 其他情况回退一天
    """
    # 解析输入的时间字符串
    trigger_datetime = datetime.strptime(trigger_time, '%Y-%m-%d %H:%M:%S')
    
    # 简化实现：直接减去1天，不考虑节假日
    previous_datetime = trigger_datetime - timedelta(days=1)
    
    # 处理周末情况，确保返回的是交易日
    if previous_datetime.weekday() == 6:  # 周日，回退到周五
        previous_datetime = previous_datetime - timedelta(days=2)
    elif previous_datetime.weekday() == 5:  # 周六，回退到周五
        previous_datetime = previous_datetime - timedelta(days=1)
    
    return previous_datetime.strftime(output_format)


def get_smart_trading_date(trigger_time: str = None, output_format: str = "%Y%m%d") -> str:
    """
    智能获取交易日：根据交易时间规则自动判断使用当天还是前一个交易日
    
    核心时间逻辑：
    - 15:30前：使用前一个交易日的数据
    - 15:30后：使用当天交易日的数据
    
    Args:
        trigger_time (str, optional): 触发时间，格式：YYYY-MM-DD HH:MM:SS
                                     如果为None则使用当前系统时间
        output_format (str): 输出日期格式，默认：%Y%m%d
    
    Returns:
        str: 智能判断的交易日，格式：YYYYMMDD
        
    业务逻辑：
        1. 解析输入时间或使用当前时间
        2. 判断是否在15:30之后
        3. 15:30后使用当天，15:30前使用前一个交易日
        4. 自动处理周末，确保返回的是交易日
    """
    # 如果没有提供trigger_time，使用当前系统时间
    if trigger_time is None:
        current_datetime = datetime.now()
    else:
        current_datetime = datetime.strptime(trigger_time, '%Y-%m-%d %H:%M:%S')
    
    # 设置15:30作为分界点（股市收盘时间）
    cutoff_time = current_datetime.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if current_datetime >= cutoff_time:
        # 15:30之后，使用当天日期（当天交易已结束）
        target_date = current_datetime
    else:
        # 15:30之前，使用前一个交易日（当天交易未结束，使用前一交易日完整数据）
        target_date = current_datetime - timedelta(days=1)
        
        # 处理周末情况，确保返回的是交易日
        if target_date.weekday() == 6:  # 周日，回退到周五
            target_date = target_date - timedelta(days=2)
        elif target_date.weekday() == 5:  # 周六，回退到周五
            target_date = target_date - timedelta(days=1)
    
    return target_date.strftime(output_format)


def get_report_date(trigger_time: str = None) -> str:
    """
    获取报告日期：生成中文格式的报告基准日期
    
    Args:
        trigger_time (str, optional): 触发时间，格式：YYYY-MM-DD HH:MM:SS
                                     如果为None则使用当前系统时间
    
    Returns:
        str: 中文格式的报告日期，格式：YYYY年MM月DD日
        
    处理流程：
        1. 调用get_smart_trading_date获取智能交易日
        2. 将日期转换为中文格式
        3. 返回适合报告使用的日期字符串
    """
    # 获取智能交易日（YYYY-MM-DD格式）
    trading_date = get_smart_trading_date(trigger_time, "%Y-%m-%d")
    
    # 转换为datetime对象以便格式化
    report_datetime = datetime.strptime(trading_date, "%Y-%m-%d")
    
    # 返回中文格式的日期
    return report_datetime.strftime("%Y年%m月%d日")


if __name__ == "__main__":
    print(get_current_datetime("2025-01-01 10:00:00"))
    print(get_previous_trading_date("2025-01-01 10:00:00"))