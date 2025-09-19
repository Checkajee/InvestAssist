#!/usr/bin/env python3
"""
综合数据分析系统
整合价格市场数据，热钱市场数据，新闻资讯数据，提供全面的市场分析
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_source.price_market_akshare import PriceMarketAkshare
from data_source.hot_money_akshare import HotMoneyAkshare
from data_source.sina_news_crawl import SinaNewsCrawl
from data_source.macro_econo import MacroEcono
from models.llm_model import GLOBAL_LLM, GLOBAL_THINKING_LLM
from utils.date_utils import get_smart_trading_date, get_report_date
from loguru import logger

class ComprehensiveMarketAnalyzer:
    """综合市场分析器"""
    
    def __init__(self):
        self.price_market = PriceMarketAkshare()
        self.hot_money = HotMoneyAkshare()
        self.sina_news = SinaNewsCrawl(start_page=1, end_page=5)  # 限制页面数以提高效率
        self.macro_econo = MacroEcono()  # 宏观经济数据源
        self.llm = GLOBAL_LLM
        self.thinking_llm = GLOBAL_THINKING_LLM
    
    async def get_comprehensive_analysis(self, trigger_time: str = None) -> Dict:
        """获取综合分析"""
        try:
            if not trigger_time:
                trigger_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            trade_date = get_smart_trading_date(trigger_time)
            logger.info(f"开始获取 {trade_date} 的综合市场分析")
            
            # 并行获取四个数据源的数据
            price_task = self.price_market.get_data(trigger_time)
            hot_money_task = self.hot_money.get_data(trigger_time)
            news_task = self.sina_news.get_data(trigger_time)
            macro_task = self.macro_econo.get_data(trigger_time)
            
            # 等待四个任务完成
            price_df, hot_money_df, news_df, macro_df = await asyncio.gather(price_task, hot_money_task, news_task, macro_task)
            
            # 构建综合分析
            analysis_result = await self._generate_comprehensive_analysis(
                trade_date, price_df, hot_money_df, news_df, macro_df
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"获取综合分析失败: {e}")
            return {
                'trade_date': trade_date,
                'comprehensive_analysis': f"分析失败: {str(e)}",
                'price_market_available': False,
                'hot_money_available': False,
                'news_available': False,
                'macro_econo_available': False,
                'data_sources_count': 0
            }
    
    async def _generate_comprehensive_analysis(self, trade_date: str, 
                                             price_df, hot_money_df, news_df, macro_df) -> Dict:
        """生成综合分析"""
        try:
            # 检查数据可用性
            price_available = not price_df.empty if price_df is not None else False
            hot_money_available = not hot_money_df.empty if hot_money_df is not None else False
            news_available = not news_df.empty if news_df is not None else False
            macro_available = not macro_df.empty if macro_df is not None else False
            
            if not price_available and not hot_money_available and not news_available and not macro_available:
                return {
                    'trade_date': trade_date,
                    'comprehensive_analysis': "当日无可用市场数据",
                    'price_market_available': False,
                    'hot_money_available': False,
                    'news_available': False,
                    'macro_econo_available': False,
                    'data_sources_count': 0
                }
            
            # 构建分析文本
            analysis_text = self._construct_comprehensive_text(
                trade_date, price_df, hot_money_df, news_df, macro_df, 
                price_available, hot_money_available, news_available, macro_available
            )
            
            # 生成LLM综合分析
            comprehensive_analysis = await self._get_llm_comprehensive_analysis(
                trade_date, analysis_text
            )
            
            return {
                'trade_date': trade_date,
                'comprehensive_analysis': comprehensive_analysis,
                'price_market_available': price_available,
                'hot_money_available': hot_money_available,
                'news_available': news_available,
                'macro_econo_available': macro_available,
                'data_sources_count': sum([price_available, hot_money_available, news_available, macro_available]),
                'raw_data': analysis_text
            }
            
        except Exception as e:
            logger.error(f"生成综合分析失败: {e}")
            return {
                'trade_date': trade_date,
                'comprehensive_analysis': f"综合分析失败: {str(e)}",
                'price_market_available': False,
                'hot_money_available': False,
                'news_available': False,
                'macro_econo_available': False,
                'data_sources_count': 0
            }
    
    def _construct_comprehensive_text(self, trade_date: str, price_df, hot_money_df, news_df, macro_df,
                                    price_available: bool, hot_money_available: bool, news_available: bool, macro_available: bool) -> str:
        """构建综合分析文本"""
        # 使用智能报告日期
        report_date = get_report_date()
        sections = [f"# {report_date} A股市场综合分析报告\n"]
        
        if macro_available:
            sections.append("## 一、宏观经济数据")
            macro_content = macro_df.iloc[0]['content'] if not macro_df.empty else "无宏观经济数据"
            sections.append(macro_content)
        
        if price_available:
            sections.append("\n## 二、价格市场数据")
            price_content = price_df.iloc[0]['content'] if not price_df.empty else "无价格市场数据"
            sections.append(price_content)
        
        if hot_money_available:
            sections.append("\n## 三、热钱市场数据")
            hot_money_content = hot_money_df.iloc[0]['content'] if not hot_money_df.empty else "无热钱市场数据"
            sections.append(hot_money_content)
        
        if news_available:
            sections.append("\n## 四、相关新闻资讯")
            # 构建新闻摘要
            news_summary = self._build_news_summary(news_df)
            sections.append(news_summary)
        
        sections.append(f"\n## 数据来源说明")
        sections.append(f"- 宏观经济数据: {'✓ 可用' if macro_available else '✗ 不可用'}")
        sections.append(f"- 价格市场数据: {'✓ 可用' if price_available else '✗ 不可用'}")
        sections.append(f"- 热钱市场数据: {'✓ 可用' if hot_money_available else '✗ 不可用'}")
        sections.append(f"- 新闻资讯数据: {'✓ 可用' if news_available else '✗ 不可用'}")
        sections.append(f"- 数据来源总数: {sum([macro_available, price_available, hot_money_available, news_available])}/4")
        
        return "\n".join(sections)
    
    def _build_news_summary(self, news_df) -> str:
        """构建新闻摘要"""
        if news_df.empty:
            return "无相关新闻资讯"
        
        summary_lines = [f"共获取 {len(news_df)} 条相关新闻："]
        
        # 按重要性排序（如果有importance字段）
        if 'importance' in news_df.columns:
            # 先填充NaN值
            news_df = news_df.copy()
            news_df['importance'] = news_df['importance'].fillna('medium')
            # 过滤掉非字符串类型的值
            news_df = news_df[news_df['importance'].apply(lambda x: isinstance(x, str))]
            
            if not news_df.empty:
                importance_order = {'high': 0, 'medium': 1, 'low': 2}
                news_df = news_df.sort_values('importance', key=lambda x: x.map(importance_order))
        
        # 显示前10条最重要的新闻
        top_news = news_df.head(10)
        
        for i, (_, row) in enumerate(top_news.iterrows(), 1):
            title = row.get('title', '无标题')
            content = row.get('content', '无内容')
            pub_time = row.get('pub_time', '')
            importance = row.get('importance', 'medium')
            
            # 确保importance是字符串类型
            if pd.isna(importance) or importance is None:
                importance = 'medium'
            else:
                importance = str(importance).lower()
            
            # 截断过长的内容
            if pd.isna(content) or content is None:
                content = '无内容'
            else:
                content = str(content)
                if len(content) > 80:
                    content = content[:80] + "..."
            
            # 确保title是字符串类型
            if pd.isna(title) or title is None:
                title = '无标题'
            else:
                title = str(title)
            
            summary_lines.append(f"{i}. 【{importance.upper()}】{title}")
            summary_lines.append(f"   {content}")
            if pub_time and not pd.isna(pub_time):
                summary_lines.append(f"   时间: {pub_time}")
            summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    async def _get_llm_comprehensive_analysis(self, trade_date: str, analysis_text: str) -> str:
        """获取LLM综合分析"""
        try:
            prompt = f"""
请基于以下{trade_date}的A股市场综合数据，生成一份专业的市场分析报告（3000字符以内）：

{analysis_text}

## 综合分析要求

请从以下维度进行综合分析：

### 1. 宏观经济环境分析
- 分析GDP、CPI、PPI等核心经济指标
- 评估就业市场和社会融资情况
- 分析进出口贸易和制造业景气度
- 判断宏观经济对股市的影响

### 2. 市场技术面分析
- 三大指数表现和技术面分析
- 市场整体走势和趋势判断
- 成交量和资金面情况

### 3. 热钱活跃度分析
- 涨停跌停股票分布和连板情况
- 龙虎榜活跃度和机构参与情况
- 概念板块热度和游资行为

### 4. 新闻资讯影响分析
- 分析重要新闻对市场的影响
- 识别政策面、基本面变化
- 评估新闻对投资者情绪的影响

### 5. 市场情绪和资金流向
- 综合判断市场情绪（乐观/悲观/中性）
- 分析主力资金和游资的流向
- 识别市场热点和冷门板块



## 输出要求
- 基于宏观经济数据与市场动态，判断当前 A 股处于牛市（慢牛 / 快牛）、熊市（慢熊 / 快熊）还是震荡市；
- 挖掘当前最具关注价值的 3-5 个板块，并列出各板块核心龙头公司；
- 结合行情与板块特征，识别潜在投资风险点，给出简要建议（仅供参考）。
- 保持客观、专业、理性
- 基于事实数据进行分析
- 请勿输出任何没有事实依据的猜测和预测
- 避免情绪化表述和绝对化判断
- 严格控制字数在3000字符以内
- 使用中文输出


请生成一份结构清晰、逻辑严谨的综合市场分析报告：
"""
            
            messages = [
                {
                    "role": "system",
                    "content": "你是一位资深的量化投资分析师和金融市场专家，具备丰富的市场分析经验。请基于多维度数据生成专业、客观、全面的市场分析报告。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await self.llm.a_run(
                messages=messages,
                temperature=0.3,
                max_tokens=3000
            )
            
            if response and response.content:
                return response.content
            else:
                return "LLM分析失败，无法生成综合分析报告"
                
        except Exception as e:
            logger.error(f"LLM综合分析失败: {e}")
            return f"综合分析生成失败: {str(e)}"
    
    async def get_quick_market_summary(self, trigger_time: str = None) -> str:
        """获取快速市场摘要"""
        try:
            if not trigger_time:
                trigger_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            trade_date = get_smart_trading_date(trigger_time)
            logger.info(f"开始获取 {trade_date} 的快速市场摘要")
            
            # 并行获取价格数据和热钱数据
            price_task = self.price_market.get_data(trigger_time)
            hot_money_task = self.hot_money.get_data(trigger_time)
            
            # 等待两个任务完成
            price_df, hot_money_df = await asyncio.gather(price_task, hot_money_task)
            
            # 检查数据可用性
            price_available = not price_df.empty if price_df is not None else False
            hot_money_available = not hot_money_df.empty if hot_money_df is not None else False
            
            if not price_available and not hot_money_available:
                return "⚠️ 当前无可用市场数据"
            
            # 构建分析文本
            analysis_text = self._construct_quick_summary_text(
                trade_date, price_df, hot_money_df, price_available, hot_money_available
            )
            
            # 生成LLM快速摘要
            quick_summary = await self._get_llm_quick_summary(
                trade_date, analysis_text
            )
            
            # 构建最终摘要
            summary = f"📊 {trade_date} 市场摘要\n"
            summary += f"数据源: {sum([price_available, hot_money_available])}/2 可用\n"
            summary += f"价格数据: {'✓' if price_available else '✗'}\n"
            summary += f"热钱数据: {'✓' if hot_money_available else '✗'}\n\n"
            summary += quick_summary
            
            return summary
            
        except Exception as e:
            logger.error(f"获取快速市场摘要失败: {e}")
            return f"获取市场摘要失败: {str(e)}"
    
    def _construct_quick_summary_text(self, trade_date: str, price_df, hot_money_df,
                                     price_available: bool, hot_money_available: bool) -> str:
        """构建快速摘要分析文本"""
        sections = [f"## {trade_date} 市场快速摘要数据\n"]
        
        if price_available:
            sections.append("### 一、价格市场数据")
            price_content = price_df.iloc[0]['content'] if not price_df.empty else "无价格市场数据"
            sections.append(price_content)
        
        if hot_money_available:
            sections.append("\n### 二、热钱市场数据")
            hot_money_content = hot_money_df.iloc[0]['content'] if not hot_money_df.empty else "无热钱市场数据"
            sections.append(hot_money_content)
        
        return "\n".join(sections)
    
    async def _get_llm_quick_summary(self, trade_date: str, analysis_text: str) -> str:
        """获取LLM快速摘要"""
        try:
            prompt = f"""
请基于以下{trade_date}的A股市场数据，生成一份简洁的市场摘要（1000字符以内）：

{analysis_text}

## 快速摘要要求

请从以下维度进行简要分析：

### 1. 市场技术面分析
- 三大指数表现和走势
- 市场整体趋势判断
- 成交量情况

### 2. 热钱活跃度分析
- 涨停跌停股票分布
- 龙虎榜活跃度
- 概念板块热度

## 输出要求
- 简洁明了，重点突出
- 基于事实数据进行分析
- 避免情绪化表述和绝对化判断
- 严格控制字数在1000字符以内
- 使用中文输出
- 格式清晰，易于阅读

请生成一份简洁的市场摘要：
"""
            
            messages = [
                {
                    "role": "system",
                    "content": "你是一位专业的市场分析师，擅长快速提取市场关键信息。请基于价格和热钱数据生成简洁、准确的市场摘要。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await self.llm.a_run(
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            if response and response.content:
                return response.content
            else:
                return "LLM分析失败，无法生成快速摘要"
                
        except Exception as e:
            logger.error(f"LLM快速摘要失败: {e}")
            return f"快速摘要生成失败: {str(e)}"

async def main():
    """测试综合分析系统"""
    print("=" * 60)
    print("🔍 综合市场分析系统测试")
    print("=" * 60)
    
    analyzer = ComprehensiveMarketAnalyzer()
    
    try:
        print("📊 正在获取综合分析...")
        analysis = await analyzer.get_comprehensive_analysis()
        
        print(f"\n📅 交易日: {analysis['trade_date']}")
        print(f"📈 数据源: {analysis['data_sources_count']}/4")
        print(f"🌍 宏观数据: {'✓' if analysis.get('macro_econo_available', False) else '✗'}")
        print(f"💰 价格数据: {'✓' if analysis['price_market_available'] else '✗'}")
        print(f"🔥 热钱数据: {'✓' if analysis['hot_money_available'] else '✗'}")
        print(f"📰 新闻数据: {'✓' if analysis.get('news_available', False) else '✗'}")
        
        print(f"\n📝 综合分析报告:")
        print("-" * 60)
        print(analysis['comprehensive_analysis'])
        print("-" * 60)
        
        print(f"\n📋 快速摘要:")
        print("-" * 60)
        summary = await analyzer.get_quick_market_summary()
        print(summary)
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
