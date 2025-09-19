"""
看跌分析师
基于四个数据源的输出进行看跌分析
"""
import pandas as pd
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_source.financial_statement_akshare import FinancialStatementAkshare
from data_source.hot_money_akshare import HotMoneyAkshare
from data_source.price_market_akshare import PriceMarketAkshare
from data_source.sina_news_crawl import SinaNewsCrawl
from data_source.stock_analysis_akshare import StockAnalysisAkshare
from models.llm_model import GLOBAL_LLM
from utils.date_utils import get_smart_trading_date


class BearAnalyst:
    """看跌分析师"""
    
    def __init__(self):
        # 数据源将由外部注入，不在此处创建
        self.financial_data = None
        self.hot_money = None
        self.price_market = None
        self.sina_news = None
        self.stock_analysis = None
        self.llm = GLOBAL_LLM
        
    async def analyze(self, trigger_time: str = None, symbol: str = "000001") -> Dict[str, Any]:
        """
        进行看跌分析
        
        Args:
            trigger_time: 触发时间
            symbol: 股票代码
            
        Returns:
            Dict: 分析结果
        """
        try:
            if not trigger_time:
                trigger_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            trade_date = get_smart_trading_date(trigger_time)
            logger.info(f"🐻 开始看跌分析: {symbol} 在 {trade_date}")
            
            # 并行获取五个数据源的数据
            financial_task = self.financial_data.get_data(trigger_time, symbol)
            hot_money_task = self.hot_money.get_data(trigger_time)
            price_task = self.price_market.get_data(trigger_time)
            news_task = self.sina_news.get_data(trigger_time)
            stock_analysis_task = self.stock_analysis.get_data(trigger_time, symbol)
            
            # 等待所有任务完成
            financial_df, hot_money_df, price_df, news_df, stock_analysis_df = await asyncio.gather(
                financial_task, hot_money_task, price_task, news_task, stock_analysis_task
            )
            
            # 提取数据内容
            financial_report = self._extract_content(financial_df, "财务数据")
            hot_money_report = self._extract_content(hot_money_df, "热钱市场数据")
            price_report = self._extract_content(price_df, "价格市场数据")
            news_report = self._extract_content(news_df, "新闻资讯")
            stock_analysis_report = self._extract_content(stock_analysis_df, "个股分析数据")
            
            # 生成看跌分析
            bear_analysis = await self._generate_bear_analysis(
                symbol, trade_date, financial_report, hot_money_report, price_report, news_report, stock_analysis_report
            )
            
            return {
                'symbol': symbol,
                'trade_date': trade_date,
                'trigger_time': trigger_time,
                'bear_analysis': bear_analysis,
                'data_sources': {
                    'financial_available': not financial_df.empty,
                    'hot_money_available': not hot_money_df.empty,
                    'price_available': not price_df.empty,
                    'news_available': not news_df.empty,
                    'stock_analysis_available': not stock_analysis_df.empty
                },
                'raw_data': {
                    'financial_report': financial_report,
                    'hot_money_report': hot_money_report,
                    'price_report': price_report,
                    'news_report': news_report,
                    'stock_analysis_report': stock_analysis_report
                }
            }
            
        except Exception as e:
            logger.error(f"🐻 看跌分析失败: {e}")
            return {
                'symbol': symbol,
                'trade_date': trade_date,
                'trigger_time': trigger_time,
                'bear_analysis': f"看跌分析失败: {str(e)}",
                'data_sources': {
                    'financial_available': False,
                    'hot_money_available': False,
                    'price_available': False,
                    'news_available': False,
                    'stock_analysis_available': False
                },
                'raw_data': {}
            }
    
    def _extract_content(self, df: pd.DataFrame, data_type: str) -> str:
        """提取DataFrame中的内容"""
        if df is None or df.empty:
            return f"{data_type}无可用数据"
        
        try:
            content = df.iloc[0]['content'] if 'content' in df.columns else str(df.iloc[0])
            return content if content else f"{data_type}内容为空"
        except Exception as e:
            logger.warning(f"提取{data_type}内容失败: {e}")
            return f"{data_type}内容提取失败"
    
    async def _generate_bear_analysis(self, symbol: str, trade_date: str, 
                                    financial_report: str, hot_money_report: str, 
                                    price_report: str, news_report: str, stock_analysis_report: str) -> str:
        """生成看跌分析"""
        try:
            # 构建分析提示词
            prompt = f"""你是一位专业的看跌分析师，负责论证不投资股票 {symbol} 的理由。

⚠️ 重要提醒：当前分析的是中国A股，所有价格和估值请使用人民币（¥）作为单位。

你的目标是提出合理的论证，强调风险、挑战和负面指标。利用提供的研究和数据来突出潜在的不利因素并有效反驳看涨论点。

请用中文回答，重点关注以下几个方面：

## 分析重点

### 1. 风险和挑战识别
- 突出市场饱和、财务不稳定或宏观经济威胁等可能阻碍股票表现的因素
- 分析行业竞争加剧、技术替代风险等外部威胁
- 识别政策风险、监管变化等系统性风险

### 2. 竞争劣势分析
- 强调市场地位较弱、创新下降或来自竞争对手威胁等脆弱性
- 分析公司护城河的薄弱环节
- 评估技术落后和产品竞争力下降的风险

### 3. 负面指标解读
- 使用财务数据、市场趋势或最近不利消息的证据来支持你的立场
- 分析估值过高、盈利能力下降等负面信号
- 识别技术面和资金面的消极变化

### 4. 市场环境不利因素
- 分析宏观经济环境对公司业务的负面影响
- 评估政策面和资金面对该股票的压力
- 识别市场情绪恶化和资金流出的风险

### 5. 投资风险提示
- 强调当前时点的投资风险
- 提供具体的风险规避建议
- 分析持有该股票可能面临的各种不利情况

## 可用数据资源

### 财务数据分析
{financial_report}

### 热钱市场数据
{hot_money_report}

### 价格市场数据
{price_report}

### 新闻资讯
{news_report}

### 个股分析数据
{stock_analysis_report}

## 输出要求

请基于以上五个数据源的信息，生成一份专业的看跌分析报告，要求：

1. **结构清晰**：按照上述5个分析重点组织内容
2. **数据支撑**：每个观点都要有具体的数据和事实支撑
3. **逻辑严密**：论证过程要逻辑清晰，结论要有说服力
4. **客观专业**：保持专业分析师的客观立场，避免过度悲观
5. **实用性强**：提供具体的风险提示和规避建议
6. **字数控制**：控制在2000字符以内，重点突出

请确保所有回答都使用中文，并基于实际数据进行分析。
"""

            messages = [
                {
                    "role": "system", 
                    "content": "你是一位资深的看跌分析师，专长于风险识别和投资风险分析。请基于实际数据生成专业的看跌分析报告。"
                },
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm.a_run(
                messages=messages,
                thinking=False,
                temperature=0.3,
                max_tokens=2000
            )
            
            if response and response.content:
                return response.content
            else:
                logger.error(f"LLM看跌分析未返回内容")
                return f"{symbol} 看跌分析生成失败"
                
        except Exception as e:
            logger.error(f"生成看跌分析失败: {e}")
            return f"{symbol} 看跌分析失败: {str(e)}"


if __name__ == "__main__":
    # 测试看跌分析师
    async def test_bear_analyst():
        analyst = BearAnalyst()
        result = await analyst.analyze("2024-08-19 09:00:00", "000001")
        print("看跌分析结果:")
        print(f"股票代码: {result['symbol']}")
        print(f"交易日: {result['trade_date']}")
        print(f"数据源可用性: {result['data_sources']}")
        print(f"\n看跌分析报告:")
        print("-" * 50)
        print(result['bear_analysis'])
    
    asyncio.run(test_bear_analyst())
