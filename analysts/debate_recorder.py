"""
辩论记录器
负责记录bear/bull两个analysts的发言情况，管理辩论流程
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from .bull_analyst import BullAnalyst
from .bear_analyst import BearAnalyst


class DebateRecorder:
    """辩论记录器"""
    
    def __init__(self):
        # 创建所有共享的数据源实例，避免重复获取数据
        from data_source.financial_statement_akshare import FinancialStatementAkshare
        from data_source.hot_money_akshare import HotMoneyAkshare
        from data_source.price_market_akshare import PriceMarketAkshare
        from data_source.sina_news_crawl import SinaNewsCrawl
        from data_source.stock_analysis_akshare import StockAnalysisAkshare
        
        # 创建共享的数据源实例
        shared_financial = FinancialStatementAkshare()
        shared_hot_money = HotMoneyAkshare()
        shared_price_market = PriceMarketAkshare()
        shared_sina_news = SinaNewsCrawl(start_page=1, end_page=5)
        shared_stock_analysis = StockAnalysisAkshare()
        
        self.bull_analyst = BullAnalyst()
        self.bear_analyst = BearAnalyst()
        
        # 让两个分析师共享所有数据源
        self.bull_analyst.financial_data = shared_financial
        self.bull_analyst.hot_money = shared_hot_money
        self.bull_analyst.price_market = shared_price_market
        self.bull_analyst.sina_news = shared_sina_news
        self.bull_analyst.stock_analysis = shared_stock_analysis
        
        self.bear_analyst.financial_data = shared_financial
        self.bear_analyst.hot_money = shared_hot_money
        self.bear_analyst.price_market = shared_price_market
        self.bear_analyst.sina_news = shared_sina_news
        self.bear_analyst.stock_analysis = shared_stock_analysis
        
        # 辩论状态
        self.debate_state = {
            "history": "",  # 完整辩论历史
            "bull_history": "",  # 看涨方历史
            "bear_history": "",  # 看跌方历史
            "current_response": "",  # 当前回应
            "count": 0,  # 发言次数
            "round": 0,  # 轮次
            "turn": "bear"  # 当前发言方：bear或bull
        }
        
        # 辩论记录
        self.debate_records = []
        
    async def conduct_debate(self, trigger_time: str = None, symbol: str = "000001") -> Dict[str, Any]:
        """
        进行两轮四次发言的辩论
        
        Args:
            trigger_time: 触发时间
            symbol: 股票代码
            
        Returns:
            Dict: 辩论结果
        """
        try:
            if not trigger_time:
                trigger_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"🗣️ 开始辩论: {symbol} 在 {trigger_time}")
            logger.info(f"辩论规则: 两轮四次发言 (Bear -> Bull -> Bear -> Bull)")
            
            # 获取初始分析结果
            initial_data = await self._get_initial_analyses(trigger_time, symbol)
            if not initial_data:
                return self._create_error_result("获取初始分析失败")
            
            # 进行两轮辩论
            for round_num in range(1, 3):  # 两轮
                logger.info(f"🔄 开始第 {round_num} 轮辩论")
                
                # 每轮两次发言
                for turn_num in range(1, 3):  # 每轮两次发言
                    if turn_num == 1:
                        # 第一轮：Bear先发言
                        speaker = "bear"
                        speaker_name = "看跌分析师"
                    else:
                        # 第二轮：Bull发言
                        speaker = "bull"
                        speaker_name = "看涨分析师"
                    
                    logger.info(f"🎤 第 {round_num} 轮第 {turn_num} 次发言: {speaker_name}")
                    
                    # 执行发言
                    response = await self._execute_speech(
                        speaker, trigger_time, symbol, initial_data
                    )
                    
                    # 记录发言
                    self._record_speech(round_num, turn_num, speaker, response)
                    
                    # 更新辩论状态
                    self._update_debate_state(speaker, response)
            
            logger.info(f"✅ 辩论完成，共进行 {self.debate_state['count']} 次发言")
            
            return {
                'symbol': symbol,
                'trigger_time': trigger_time,
                'debate_completed': True,
                'total_speeches': self.debate_state['count'],
                'debate_state': self.debate_state,
                'debate_records': self.debate_records,
                'initial_data': initial_data
            }
            
        except Exception as e:
            logger.error(f"🗣️ 辩论失败: {e}")
            return self._create_error_result(f"辩论失败: {str(e)}")
    
    async def _get_initial_analyses(self, trigger_time: str, symbol: str) -> Optional[Dict[str, Any]]:
        """获取初始分析结果"""
        try:
            # 并行获取bull和bear分析师的完整分析结果
            bull_analysis_task = self.bull_analyst.analyze(trigger_time, symbol)
            bear_analysis_task = self.bear_analyst.analyze(trigger_time, symbol)
            
            # 等待分析完成
            bull_result, bear_result = await asyncio.gather(
                bull_analysis_task, bear_analysis_task
            )
            
            # 提取分析内容
            initial_data = {
                'bull_analysis': bull_result.get('bull_analysis', '看涨分析获取失败'),
                'bear_analysis': bear_result.get('bear_analysis', '看跌分析获取失败'),
                'bull_data_sources': bull_result.get('data_sources', {}),
                'bear_data_sources': bear_result.get('data_sources', {}),
                'symbol': symbol,
                'trigger_time': trigger_time,
                'trade_date': bull_result.get('trade_date', '未知交易日')
            }
            
            logger.info(f"✅ 获取初始分析完成: {symbol}")
            return initial_data
            
        except Exception as e:
            logger.error(f"获取初始分析失败: {e}")
            return None
    
    async def _execute_speech(self, speaker: str, trigger_time: str, symbol: str, 
                            initial_data: Dict[str, Any]) -> str:
        """执行发言"""
        try:
            if speaker == "bear":
                # 看跌分析师发言
                analysis = await self._generate_bear_speech(
                    symbol, trigger_time, initial_data
                )
            else:
                # 看涨分析师发言
                analysis = await self._generate_bull_speech(
                    symbol, trigger_time, initial_data
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"{speaker} 发言失败: {e}")
            return f"{speaker} 发言失败: {str(e)}"
    
    async def _generate_bear_speech(self, symbol: str, trigger_time: str, 
                                  initial_data: Dict[str, Any]) -> str:
        """生成看跌分析师发言"""
        try:
            # 获取看跌分析师的基础分析
            bear_base_analysis = initial_data.get('bear_analysis', '看跌分析获取失败')
            
            # 构建发言提示词，基于已有的分析结果进行辩论
            prompt = f"""你是一位专业的看跌分析师，正在进行投资辩论。

⚠️ 重要提醒：当前分析的是中国A股，所有价格和估值请使用人民币（¥）作为单位。

🎯 **核心信息**：
- **当前分析的股票代码：{symbol}**
- **请务必在发言中明确提及股票代码 {symbol}**
- 当前轮次：第 {self.debate_state['round']} 轮
- 发言次数：第 {self.debate_state['count'] + 1} 次

## 你的基础分析
{bear_base_analysis}

## 辩论历史
{self.debate_state['history']}

## 看涨方最新观点
{self.debate_state['current_response'] if 'Bull' in self.debate_state['current_response'] else '暂无'}

## 发言要求

请基于你的基础分析进行专业辩论发言，要求：

1. **股票代码明确**：**必须在发言开头明确提及股票代码 {symbol}**
2. **基于分析**：基于上述你的基础分析结果进行发言
3. **针对性强**：如果这是回应发言，要直接回应看涨方的观点
4. **逻辑严密**：论证过程要逻辑清晰，结论要有说服力
5. **风险导向**：重点关注投资风险和负面因素
6. **对话风格**：以自然对话的方式呈现，不要使用特殊格式
7. **字数控制**：控制在800字符以内

⚠️ **重要提醒**：请确保使用中文，基于你的分析结果进行辩论，并且始终围绕股票代码 {symbol} 进行讨论。"""

            messages = [
                {
                    "role": "system", 
                    "content": "你是一位资深的看跌分析师，正在进行投资辩论。请基于你的分析结果生成专业的看跌论点。"
                },
                {"role": "user", "content": prompt}
            ]
            
            response = await self.bull_analyst.llm.a_run(
                messages=messages,
                thinking=False,
                temperature=0.3,
                max_tokens=800
            )
            
            if response and response.content:
                return f"Bear Analyst: {response.content}"
            else:
                return f"Bear Analyst: 发言生成失败"
                
        except Exception as e:
            logger.error(f"生成看跌发言失败: {e}")
            return f"Bear Analyst: 发言失败: {str(e)}"
    
    async def _generate_bull_speech(self, symbol: str, trigger_time: str, 
                                  initial_data: Dict[str, Any]) -> str:
        """生成看涨分析师发言"""
        try:
            # 获取看涨分析师的基础分析
            bull_base_analysis = initial_data.get('bull_analysis', '看涨分析获取失败')
            
            # 构建发言提示词，基于已有的分析结果进行辩论
            prompt = f"""你是一位专业的看涨分析师，正在进行投资辩论。

⚠️ 重要提醒：当前分析的是中国A股，所有价格和估值请使用人民币（¥）作为单位。

🎯 **核心信息**：
- **当前分析的股票代码：{symbol}**
- **请务必在发言中明确提及股票代码 {symbol}**
- 当前轮次：第 {self.debate_state['round']} 轮
- 发言次数：第 {self.debate_state['count'] + 1} 次

## 你的基础分析
{bull_base_analysis}

## 辩论历史
{self.debate_state['history']}

## 看跌方最新观点
{self.debate_state['current_response'] if 'Bear' in self.debate_state['current_response'] else '暂无'}

## 发言要求

请基于你的基础分析进行专业辩论发言，要求：

1. **股票代码明确**：**必须在发言开头明确提及股票代码 {symbol}**
2. **基于分析**：基于上述你的基础分析结果进行发言
3. **针对性强**：如果这是回应发言，要直接回应看跌方的观点
4. **逻辑严密**：论证过程要逻辑清晰，结论要有说服力
5. **机会导向**：重点关注投资机会和积极因素
6. **对话风格**：以自然对话的方式呈现，不要使用特殊格式
7. **字数控制**：控制在800字符以内

⚠️ **重要提醒**：请确保使用中文，基于你的分析结果进行辩论，并且始终围绕股票代码 {symbol} 进行讨论。"""

            messages = [
                {
                    "role": "system", 
                    "content": "你是一位资深的看涨分析师，正在进行投资辩论。请基于你的分析结果生成专业的看涨论点。"
                },
                {"role": "user", "content": prompt}
            ]
            
            response = await self.bull_analyst.llm.a_run(
                messages=messages,
                thinking=False,
                temperature=0.3,
                max_tokens=800
            )
            
            if response and response.content:
                return f"Bull Analyst: {response.content}"
            else:
                return f"Bull Analyst: 发言生成失败"
                
        except Exception as e:
            logger.error(f"生成看涨发言失败: {e}")
            return f"Bull Analyst: 发言失败: {str(e)}"
    
    def _record_speech(self, round_num: int, turn_num: int, speaker: str, response: str):
        """记录发言"""
        record = {
            'round': round_num,
            'turn': turn_num,
            'speaker': speaker,
            'response': response,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.debate_records.append(record)
        logger.info(f"📝 记录发言: 第{round_num}轮第{turn_num}次 - {speaker}")
    
    def _update_debate_state(self, speaker: str, response: str):
        """更新辩论状态"""
        self.debate_state['history'] += "\n" + response if self.debate_state['history'] else response
        
        if speaker == "bear":
            self.debate_state['bear_history'] += "\n" + response if self.debate_state['bear_history'] else response
        else:
            self.debate_state['bull_history'] += "\n" + response if self.debate_state['bull_history'] else response
        
        self.debate_state['current_response'] = response
        self.debate_state['count'] += 1
        
        # 更新轮次
        if self.debate_state['count'] % 2 == 0:
            self.debate_state['round'] += 1
        
        # 更新当前发言方
        self.debate_state['turn'] = "bull" if speaker == "bear" else "bear"
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'symbol': 'unknown',
            'trigger_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'debate_completed': False,
            'total_speeches': 0,
            'debate_state': self.debate_state,
            'debate_records': self.debate_records,
            'initial_data': {},
            'error': error_msg
        }


if __name__ == "__main__":
    # 测试辩论记录器
    async def test_debate_recorder():
        recorder = DebateRecorder()
        result = await recorder.conduct_debate("2024-08-19 09:00:00", "000001")
        
        print("辩论结果:")
        print(f"股票代码: {result['symbol']}")
        print(f"辩论完成: {result['debate_completed']}")
        print(f"总发言次数: {result['total_speeches']}")
        print(f"辩论记录数量: {len(result['debate_records'])}")
        
        print("\n辩论历史:")
        print("-" * 50)
        print(result['debate_state']['history'])
    
    asyncio.run(test_debate_recorder())
