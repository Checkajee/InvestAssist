"""
基于 akshare 的财务数据源
整合财务报表数据，生成综合财务分析
"""
import pandas as pd
import asyncio
import traceback
from datetime import datetime
from .data_source_base import DataSourceBase
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.akshare_utils import akshare_cached
from models.llm_model import GLOBAL_LLM, GLOBAL_VISION_LLM
from loguru import logger
from config.config import cfg
from utils.date_utils import get_smart_trading_date
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from typing import Dict, Any, Optional

class FinancialStatementAkshare(DataSourceBase):
    def __init__(self):
        super().__init__("financial_statement_akshare")
        
    async def get_data(self, trigger_time: str, symbol: str) -> pd.DataFrame:
        """
        获取单个股票的财务数据
        
        Args:
            trigger_time: 触发时间
            symbol: 股票代码 (必需)
        """
        try:
            if not symbol:
                logger.error("股票代码不能为空")
                return pd.DataFrame()
                
            cache_key = f"{trigger_time}_{symbol}"
            df = self.get_data_cached(cache_key)
            if df is not None:
                return df
            
            trade_date = get_smart_trading_date(trigger_time)     
            logger.info(f"获取 {symbol} 在 {trade_date} 的财务数据")
            
            llm_summary_dict = await self.get_stock_financial_analysis(symbol, trade_date)
            data = [{
                "title": f"{symbol} {trade_date}:财务数据分析",
                "content": llm_summary_dict["llm_summary"],
                "pub_time": trigger_time,
                "url": None,
                "symbol": symbol
            }]
            
            df = pd.DataFrame(data)
            self.save_data_cached(cache_key, df)
            return df
                
        except Exception as e:
            logger.error(f"获取财务数据失败: {e}")
            return pd.DataFrame()
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """
        将6位数字股票代码转换为带前缀/后缀的格式
        
        Args:
            symbol: 6位数字股票代码，如 "000001", "600519", "301389"
            
        Returns:
            str: 带前缀/后缀的股票代码，如 "SH600519", "301389.SZ"
        """
        if not symbol or len(symbol) != 6 or not symbol.isdigit():
            return symbol
            
        # 根据股票代码规则判断市场
        if symbol.startswith(('600', '601', '603', '605', '688')):
            # 上海主板、科创板
            return f"SH{symbol}"
        elif symbol.startswith(('000', '001', '002', '003')):
            # 深圳主板、中小板
            return f"{symbol}.SZ"
        elif symbol.startswith('300'):
            # 创业板
            return f"{symbol}.SZ"
        elif symbol.startswith('301'):
            # 创业板
            return f"{symbol}.SZ"
        else:
            # 默认使用深圳格式
            return f"{symbol}.SZ"
    
    def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取股票财务数据
        
        Args:
            symbol: 股票代码 (6位数字)
            
        Returns:
            Dict: 包含主要财务指标的财务数据
        """
        try:
            logger.info(f"🔍 开始获取{symbol}的财务数据")
            
            # 转换股票代码格式
            formatted_symbol = self._convert_symbol_format(symbol)
            logger.debug(f"📊 股票代码格式转换: {symbol} -> {formatted_symbol}")
            
            financial_data = {}
            
            # 1. 获取财务摘要指标 (替代失效的stock_financial_analysis_indicator_em)
            try:
                logger.debug(f"📊 尝试获取{symbol}财务摘要指标...")
                main_indicators = akshare_cached.run(
                    func_name="stock_financial_abstract",
                    func_kwargs={"symbol": symbol},  # 使用原始6位代码
                    verbose=False
                )
                if main_indicators is not None and not main_indicators.empty:
                    financial_data['main_indicators'] = main_indicators
                    logger.info(f"✅ 成功获取{symbol}财务摘要指标: {len(main_indicators)}条记录")
                    logger.debug(f"财务摘要指标列名: {list(main_indicators.columns)}")
                else:
                    logger.warning(f"⚠️ {symbol}财务摘要指标为空")
            except Exception as e:
                logger.warning(f"❌ 获取{symbol}财务摘要指标失败: {e}")
            
            
            # 记录最终结果
            if financial_data:
                logger.info(f"✅ 财务数据获取完成: {symbol} (格式: {formatted_symbol}), 包含{len(financial_data)}个数据集")
                for key, value in financial_data.items():
                    if hasattr(value, '__len__'):
                        logger.info(f"  - {key}: {len(value)}条记录")
            else:
                logger.warning(f"⚠️ 未能获取{symbol}的任何财务数据")
            
            return financial_data
            
        except Exception as e:
            logger.error(f"❌ 获取{symbol}财务数据失败: {e}")
            return {}
    
    
    def _find_column_by_keywords(self, df: pd.DataFrame, keywords: list) -> str:
        """
        根据关键词查找列名
        
        Args:
            df: DataFrame
            keywords: 关键词列表，按优先级排序
            
        Returns:
            str: 找到的列名，如果没找到返回None
        """
        for keyword in keywords:
            for col in df.columns:
                if keyword.lower() in col.lower():
                    return col
        return None
    
    def format_financial_data(self, financial_data: Dict[str, Any], symbol: str) -> str:
        """
        格式化财务摘要数据为文本
        """
        if not financial_data:
            return f"{symbol} 无财务数据"
        
        formatted_text = f"📊 {symbol} 财务摘要数据:\n\n"
        
        # 财务摘要指标
        if 'main_indicators' in financial_data:
            main_indicators = financial_data['main_indicators']
            if not main_indicators.empty:
                formatted_text += "## 财务摘要指标:\n\n"
                
                # 直接展示财务摘要表格的关键部分
                # 只显示前20行，避免数据过多
                display_rows = main_indicators.head(20)
                
                # 获取最新的几个报告期列
                date_columns = [col for col in main_indicators.columns if col.startswith('20')]
                recent_periods = date_columns[:4] if len(date_columns) >= 4 else date_columns
                
                if recent_periods:
                    # 创建显示列：指标 + 最近几个报告期
                    display_columns = ['指标'] + recent_periods
                    display_data = display_rows[display_columns]
                    
                    # 格式化显示
                    for _, row in display_data.iterrows():
                        try:
                            indicator = row['指标']
                            formatted_text += f"**{indicator}:**\n"
                            
                            for period in recent_periods:
                                value = row[period]
                                if pd.notna(value) and value != 0:
                                    # 根据指标类型决定显示格式
                                    if '每股' in indicator or '率' in indicator:
                                        formatted_text += f"  {period}: {value:.4f}\n"
                                    else:
                                        formatted_text += f"  {period}: {value:.2f}\n"
                                else:
                                    formatted_text += f"  {period}: 无数据\n"
                            formatted_text += "\n"
                        except Exception as e:
                            logger.warning(f"格式化行数据失败: {e}")
                            continue
                else:
                    formatted_text += "  无可用报告期数据\n\n"
        
        return formatted_text
    
    async def get_stock_financial_analysis(self, symbol: str, trade_date: str) -> dict:
        """
        获取单个股票的财务分析
        """
        try:
            logger.info(f"获取 {symbol} 在 {trade_date} 的财务分析")
            
            # 获取财务数据
            financial_data = self.get_financial_data(symbol)
            
            if not financial_data:
                return {
                    'trade_date': trade_date,
                    'symbol': symbol,
                    'raw_data': "无财务数据",
                    'llm_summary': f"{symbol} 在 {trade_date} 无可用财务数据",
                    'data_count': 0
                }
            
            # 格式化财务数据
            formatted_data = self.format_financial_data(financial_data, symbol)
            
            # 构建LLM分析提示词
            prompt = f"""
请分析以下{symbol}股票的财务摘要数据，并给出专业的财务分析报告（1500字符以内）：

## 财务摘要数据
{formatted_data}

## 分析要求
请基于企业的财务数据，进行以下维度的分析：​
1. 盈利能力：分析企业获取利润的能力，包括但不限于毛利率、净利率、净资产收益率等相关指标的表现及变动趋势。​
2. 营运能力：评估企业资产管理的效率，如存货周转率、应收账款周转率、总资产周转率等指标反映出的运营状况。​
3. 偿债能力：从短期和长期两个层面分析企业偿还债务的能力，涉及流动比率、速动比率、资产负债率、利息保障倍数等指标。​
4. 成长能力：探究企业的发展潜力，通过营业收入增长率、净利润增长率、总资产增长率等指标分析其成长趋势。​
5. 财务健康状况：综合上述盈利能力、营运能力、偿债能力等方面，判断企业整体的财务健康程度。​
6. 每股净资产：解读每股净资产的数值及变动情况，分析其反映的股东权益状况。​
7. 经营现金流量：分析经营活动产生的现金流量净额等相关数据，评估企业经营活动的现金获取能力和现金周转情况。

## 输出要求
- 基于实际财务摘要数据进行分析
- 保持客观专业，避免主观判断
- 突出关键财务指标和变化趋势
- 控制在2000字符以内
"""
            
            messages = [
                {
                    "role": "system", 
                    "content": "你是一位资深的财务分析师，专长于财务报表分析和企业财务评估。请基于实际财务数据生成专业的财务分析报告。"
                },
                {"role": "user", "content": prompt}
            ]
            
            response = await GLOBAL_LLM.a_run(
                messages=messages,
                thinking=False,
                temperature=0.3,
                max_tokens=1500
            )
            
            if response and response.content:
                llm_summary = response.content
            else:
                logger.error(f"LLM财务分析未返回内容")
                llm_summary = f"{symbol} 财务分析失败"
            
            return {
                'trade_date': trade_date,
                'symbol': symbol,
                'raw_data': formatted_data,
                'llm_summary': llm_summary,
                'data_count': len(financial_data)
            }
                
        except Exception as e:
            traceback.print_exc()
            logger.error(f"获取{symbol}财务分析失败: {e}")
            return {
                'trade_date': trade_date,
                'symbol': symbol,
                'raw_data': "数据获取失败",
                'llm_summary': f"{symbol} 财务分析失败: {str(e)}",
                'data_count': 0
            }
    

if __name__ == "__main__":
    # 测试单个股票财务数据
    financial_analyzer = FinancialStatementAkshare()
    df = asyncio.run(financial_analyzer.get_data("2024-08-19 09:00:00", "000001"))
    print(df.content.values[0] if not df.empty else "无数据")
