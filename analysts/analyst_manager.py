"""
分析师管理器
接收两轮辩论双方的四次发言，进行分析总结，用于最终做出决策
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.llm_model import GLOBAL_LLM
from .debate_recorder import DebateRecorder
from utils.date_utils import get_smart_trading_date


class AnalystManager:
    """分析师管理器"""
    
    def __init__(self):
        self.llm = GLOBAL_LLM
        self.debate_recorder = DebateRecorder()
        
    async def conduct_full_analysis(self, trigger_time: str = None, symbol: str = "000001") -> Dict[str, Any]:
        """
        进行完整的分析流程：辩论 -> 决策
        
        Args:
            trigger_time: 触发时间
            symbol: 股票代码
            
        Returns:
            Dict: 完整分析结果
        """
        try:
            if not trigger_time:
                trigger_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"🎯 开始完整分析流程: {symbol} 在 {trigger_time}")
            
            # 第一步：进行辩论
            logger.info("📢 第一步：获取数据并进行分析师辩论")
            debate_result = await self.debate_recorder.conduct_debate(trigger_time, symbol)
            
            if not debate_result.get('debate_completed', False):
                return self._create_error_result("辩论未完成", debate_result)
            
            # 第二步：基于辩论结果做出最终决策
            logger.info("⚖️ 第二步：基于辩论结果做出最终决策")
            decision_result = await self._make_final_decision(
                symbol, trigger_time, debate_result
            )
            
            # 整合结果
            final_result = {
                'symbol': symbol,
                'trigger_time': trigger_time,
                'analysis_completed': True,
                'debate_result': debate_result,
                'decision_result': decision_result,
                'summary': {
                    'total_speeches': debate_result['total_speeches'],
                    'debate_rounds': 2,
                    'final_decision': decision_result.get('investment_decision', '未知'),
                    'confidence_level': decision_result.get('confidence_level', '未知')
                }
            }
            
            logger.info(f"✅ 完整分析流程完成: {symbol}")
            return final_result
            
        except Exception as e:
            logger.error(f"🎯 完整分析流程失败: {e}")
            return self._create_error_result(f"分析流程失败: {str(e)}", {})
    
    async def _make_final_decision(self, symbol: str, trigger_time: str, 
                                 debate_result: Dict[str, Any]) -> Dict[str, Any]:
        """基于辩论结果做出最终决策"""
        try:
            # 提取辩论数据
            debate_history = debate_result['debate_state']['history']
            bull_history = debate_result['debate_state']['bull_history']
            bear_history = debate_result['debate_state']['bear_history']
            initial_data = debate_result['initial_data']
            
            # 构建决策提示词
            prompt = f"""作为投资组合经理和辩论主持人，您的职责是批判性地评估这轮辩论并做出明确决策：支持看跌分析师、看涨分析师，或者仅在基于所提出论点有强有力理由时选择持有。

简洁地总结双方的关键观点，重点关注最有说服力的证据或推理。您的建议——买入、卖出或持有——必须明确且可操作。避免仅仅因为双方都有有效观点就默认选择持有；要基于辩论中最强有力的论点做出承诺。

此外，为交易员制定详细的投资计划。这应该包括：

您的建议：基于最有说服力论点的明确立场。
理由：解释为什么这些论点导致您的结论。
战略行动：实施建议的具体步骤。

📊 目标价格分析：基于所有可用报告（个股财报数据、财经新闻、市场情绪、个股分析数据、市场数据），提供全面的目标价格区间和具体价格目标。考虑：
- 当前价格基准：首先确认股票的最新收盘价格作为分析基准
- 大盘行情及财经新闻对整体市场的影响
- 个股分析数据中的估值（选择TTM估值）
- 个股财报数据对价格预期的影响
- 个股新闻对价格预期的影响
- 情绪驱动的价格调整
- 技术支撑/阻力位
- 风险调整价格情景（保守、基准、乐观）
- 价格目标的时间范围（1个月、3个月、6个月）

💰 **目标价格要求**：
1. 必须提供具体的目标价格 - 不要回复"无法确定"或"需要更多信息"
2. **真实价格基准**：必须从个股分析数据中提取真实的"最新收盘价"作为分析基准，不得假设或估算
3. 价格合理性检查：确保目标价格基于真实当前价格合理推算，避免目标价格低于当前价格
4. 价格标注：明确标注从数据中提取的真实当前价格，并说明目标价格的计算逻辑
5. 时间对应：短期（1个月）目标应接近当前价格，中长期目标体现合理增长预期
6. **禁止假设**：严禁使用"假设为XX元"、"基于一般估值"等表述，必须使用数据中的真实价格

考虑您在类似情况下的过去错误。利用这些见解来完善您的决策制定，确保您在学习和改进。以对话方式呈现您的分析，就像自然说话一样，不使用特殊格式。

## 股票信息
- 股票代码：{symbol}
- 分析时间：{trigger_time}
- 当前日期：{datetime.now().strftime('%Y年%m月%d日')}
- 交易日：{get_smart_trading_date(trigger_time, '%Y年%m月%d日')}

⚠️ **重要提醒**：
1. **当前价格基准**：请从提供的个股分析数据中提取并明确标注当前股票的最新收盘价格（不要假设或估算价格）
2. **价格数据来源**：个股分析数据中包含"最新收盘价"信息，请务必使用这个真实数据作为分析基准
3. **目标价格合理性**：目标价格必须基于真实当前价格进行合理推算，不能出现目标价格低于当前价格的不合理情况
4. **数据时效性**：请基于最新数据进行投资决策，不要引用过时的历史价格数据
5. **价格验证**：如果数据中包含历史价格信息，请确保在投资建议中明确指出这是历史参考数据，并基于当前市场情况提供投资建议
6. **目标价格逻辑**：短期目标价格应接近当前价格，中长期目标价格应体现合理的增长预期
7. **禁止假设价格**：绝对不要使用"假设为XX元"或"基于一般估值水平"等表述，必须使用数据中提供的真实价格

## 综合分析报告

### 市场研究（价格市场数据）
{initial_data.get('price_report', '无数据')}

### 情绪分析（热钱市场数据）
{initial_data.get('hot_money_report', '无数据')}

### 新闻分析
{initial_data.get('news_report', '无数据')}

### 基本面分析（财务数据）
{initial_data.get('financial_report', '无数据')}

### 个股分析数据
{initial_data.get('stock_analysis_report', '无数据')}

## 辩论历史

### 完整辩论记录
{debate_history}

### 看涨方观点汇总
{bull_history}

### 看跌方观点汇总
{bear_history}

## 决策要求

请基于以上信息做出最终投资决策，包括：

1. **投资决策**：明确选择买入、卖出或持有
2. **决策理由**：基于辩论中最强有力的论点
3. **目标价格**：提供具体的价格目标（短期、中期、长期）
4. **风险提示**：识别主要风险和应对策略
5. **实施计划**：具体的操作建议和时间安排
6. **信心水平**：对决策的信心程度（高/中/低）

请用中文撰写所有分析内容和建议。"""

            messages = [
                {
                    "role": "system", 
                    "content": "你是一位资深的投资组合经理，专长于投资决策和风险管理。请基于辩论结果生成专业的投资决策报告。"
                },
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm.a_run(
                messages=messages,
                thinking=False,
                temperature=0.3,
                max_tokens=3000
            )
            
            if response and response.content:
                # 解析决策结果
                decision_analysis = response.content
                
                # 提取关键信息
                investment_decision = self._extract_decision(decision_analysis)
                confidence_level = self._extract_confidence(decision_analysis)
                target_price = self._extract_target_price(decision_analysis)
                
                return {
                    'investment_decision': investment_decision,
                    'confidence_level': confidence_level,
                    'target_price': target_price,
                    'decision_analysis': decision_analysis,
                    'debate_summary': {
                        'bull_key_points': self._extract_key_points(bull_history),
                        'bear_key_points': self._extract_key_points(bear_history),
                        'winning_arguments': self._extract_winning_arguments(decision_analysis)
                    }
                }
            else:
                logger.error(f"LLM决策分析未返回内容")
                return self._create_decision_error("决策分析生成失败")
                
        except Exception as e:
            logger.error(f"生成最终决策失败: {e}")
            return self._create_decision_error(f"决策生成失败: {str(e)}")
    
    def _extract_decision(self, analysis: str) -> str:
        """从分析中提取投资决策"""
        analysis_lower = analysis.lower()
        if '买入' in analysis_lower or 'buy' in analysis_lower:
            return '买入'
        elif '卖出' in analysis_lower or 'sell' in analysis_lower:
            return '卖出'
        elif '持有' in analysis_lower or 'hold' in analysis_lower:
            return '持有'
        else:
            return '持有'  # 默认持有
    
    def _extract_confidence(self, analysis: str) -> str:
        """从分析中提取信心水平"""
        import re
        
        # 使用正则表达式精确匹配信心水平表述
        # 匹配格式如：信心水平: 高、信心水平：中等、🎯 **信心水平**: 高 等
        confidence_patterns = [
            r'信心水平[：:]\s*高',
            r'🎯\s*\*\*信心水平\*\*[：:]\s*高',
            r'信心水平[：:]\s*中等?',
            r'🎯\s*\*\*信心水平\*\*[：:]\s*中等?',
            r'信心水平[：:]\s*低',
            r'🎯\s*\*\*信心水平\*\*[：:]\s*低'
        ]
        
        for i, pattern in enumerate(confidence_patterns):
            if re.search(pattern, analysis, re.IGNORECASE):
                if i < 2:  # 前两个模式匹配"高"
                    return '高'
                elif i < 4:  # 中间两个模式匹配"中等"
                    return '中'
                else:  # 后两个模式匹配"低"
                    return '低'
        
        # 如果没有找到明确的信心水平表述，使用简单的关键词匹配
        analysis_lower = analysis.lower()
        if '信心水平高' in analysis_lower or '信心高' in analysis_lower:
            return '高'
        elif '信心水平中等' in analysis_lower or '信心中等' in analysis_lower or '信心水平中' in analysis_lower:
            return '中'
        elif '信心水平低' in analysis_lower or '信心低' in analysis_lower:
            return '低'
        else:
            return '中'  # 默认中等
    
    def _extract_target_price(self, analysis: str) -> Dict[str, str]:
        """从分析中提取目标价格"""
        import re
        
        # 默认值
        target_price = {
            'short_term': '待分析',
            'medium_term': '待分析',
            'long_term': '待分析'
        }
        
        if not analysis:
            return target_price
        
        # 尝试提取短期目标价格（1个月）
        short_patterns = [
            r'短期.*?(\d+\.?\d*)元',
            r'1个月.*?(\d+\.?\d*)元',
            r'近期.*?(\d+\.?\d*)元',
            r'短期目标.*?(\d+\.?\d*)',
            r'1个月目标.*?(\d+\.?\d*)'
        ]
        
        # 尝试提取中期目标价格（3个月）
        medium_patterns = [
            r'中期.*?(\d+\.?\d*)元',
            r'3个月.*?(\d+\.?\d*)元',
            r'中期目标.*?(\d+\.?\d*)',
            r'3个月目标.*?(\d+\.?\d*)'
        ]
        
        # 尝试提取长期目标价格（6个月）
        long_patterns = [
            r'长期.*?(\d+\.?\d*)元',
            r'6个月.*?(\d+\.?\d*)元',
            r'长期目标.*?(\d+\.?\d*)',
            r'6个月目标.*?(\d+\.?\d*)'
        ]
        
        # 通用价格模式
        price_patterns = [
            r'(\d+\.?\d*)元',
            r'(\d+\.?\d*)块',
            r'价格.*?(\d+\.?\d*)',
            r'目标.*?(\d+\.?\d*)'
        ]
        
        # 提取短期价格
        for pattern in short_patterns:
            match = re.search(pattern, analysis, re.IGNORECASE)
            if match:
                target_price['short_term'] = f"{match.group(1)}元"
                break
        
        # 提取中期价格
        for pattern in medium_patterns:
            match = re.search(pattern, analysis, re.IGNORECASE)
            if match:
                target_price['medium_term'] = f"{match.group(1)}元"
                break
        
        # 提取长期价格
        for pattern in long_patterns:
            match = re.search(pattern, analysis, re.IGNORECASE)
            if match:
                target_price['long_term'] = f"{match.group(1)}元"
                break
        
        # 如果没有找到具体的时间段，尝试提取所有价格
        if all(v == '待分析' for v in target_price.values()):
            prices = []
            for pattern in price_patterns:
                matches = re.findall(pattern, analysis, re.IGNORECASE)
                prices.extend(matches)
            
            if prices:
                # 取前三个价格作为短期、中期、长期目标
                prices = list(set(prices))  # 去重
                if len(prices) >= 1:
                    target_price['short_term'] = f"{prices[0]}元"
                if len(prices) >= 2:
                    target_price['medium_term'] = f"{prices[1]}元"
                if len(prices) >= 3:
                    target_price['long_term'] = f"{prices[2]}元"
        
        return target_price
    
    def _extract_key_points(self, history: str) -> List[str]:
        """提取关键观点"""
        if not history:
            return []
        
        # 简单的关键点提取，实际可以更复杂
        points = []
        lines = history.split('\n')
        for line in lines:
            if line.strip() and len(line.strip()) > 20:
                points.append(line.strip())
        
        return points[:3]  # 返回前3个关键点
    
    def _extract_winning_arguments(self, analysis: str) -> List[str]:
        """提取获胜论点"""
        if not analysis:
            return []
        
        # 简单的获胜论点提取
        arguments = []
        lines = analysis.split('\n')
        for line in lines:
            if '关键' in line or '重要' in line or '主要' in line:
                arguments.append(line.strip())
        
        return arguments[:3]  # 返回前3个获胜论点
    
    def _create_error_result(self, error_msg: str, debate_result: Dict[str, Any]) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'symbol': 'unknown',
            'trigger_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'analysis_completed': False,
            'debate_result': debate_result,
            'decision_result': {},
            'error': error_msg
        }
    
    def _create_decision_error(self, error_msg: str) -> Dict[str, Any]:
        """创建决策错误结果"""
        return {
            'investment_decision': '持有',
            'confidence_level': '低',
            'target_price': {'short_term': '未知', 'medium_term': '未知', 'long_term': '未知'},
            'decision_analysis': f"决策分析失败: {error_msg}",
            'debate_summary': {
                'bull_key_points': [],
                'bear_key_points': [],
                'winning_arguments': []
            }
        }


if __name__ == "__main__":
    # 测试分析师管理器
    async def test_analyst_manager():
        manager = AnalystManager()
        result = await manager.conduct_full_analysis("2024-08-19 09:00:00", "000001")
        
        print("完整分析结果:")
        print(f"股票代码: {result['symbol']}")
        print(f"分析完成: {result['analysis_completed']}")
        
        if result['analysis_completed']:
            print(f"投资决策: {result['decision_result']['investment_decision']}")
            print(f"信心水平: {result['decision_result']['confidence_level']}")
            print(f"总发言次数: {result['summary']['total_speeches']}")
            
            print("\n决策分析:")
            print("-" * 50)
            print(result['decision_result']['decision_analysis'])
        else:
            print(f"分析失败: {result.get('error', '未知错误')}")
    
    asyncio.run(test_analyst_manager())
