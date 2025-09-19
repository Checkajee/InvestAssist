#!/usr/bin/env python3
"""
交互式LLM聊天程序
"""
import asyncio
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.llm_model import GLOBAL_LLM, GLOBAL_THINKING_LLM
from config.config import cfg
from comprehensive_analysis import ComprehensiveMarketAnalyzer
from analysts.analyst_manager import AnalystManager
from loguru import logger

class TradingChatBot:
    def __init__(self):
        self.llm = GLOBAL_LLM
        self.thinking_llm = GLOBAL_THINKING_LLM
        self.conversation_history = []
        self.market_analyzer = ComprehensiveMarketAnalyzer()
        self.analyst_manager = AnalystManager()
        
    async def chat(self, user_input: str, use_thinking: bool = False) -> str:
        """与LLM进行对话"""
        start_time = time.time()
        
        try:
            # 检查是否是股票/市场相关的问题，并确定具体的数据需求
            market_data = None
            data_source_type = None
            
            # 价格相关关键词
            price_keywords = ['大盘', '指数', '上证', '深证', '创业板', '科创', '走势', 'K线', '技术分析', '涨跌幅']
            # 热钱相关关键词
            hot_money_keywords = ['涨停', '跌停', '股票', '龙虎榜', '资金', '概念', '板块', '游资', '机构']
            # 新闻相关关键词
            news_keywords = ['新闻', '消息', '资讯', '公告', '政策']
            # 财务相关关键词
            financial_keywords = ['财务', '财报', '资产负债表', '利润表', '现金流', '营业收入', '净利润', '资产', '负债', '财务分析', '财务数据']
            # 综合市场相关关键词
            market_keywords = ['市场', '行情', '分析', '总结', '概况']
            
            if any(keyword in user_input for keyword in price_keywords):
                data_start = time.time()
                market_data = await self.get_price_data()
                data_time = time.time() - data_start
                data_source_type = "价格数据"
                logger.info(f"📊 获取{data_source_type}耗时: {data_time:.2f}秒")
            
            elif any(keyword in user_input for keyword in hot_money_keywords):
                data_start = time.time()
                market_data = await self.get_hot_money_data()
                data_time = time.time() - data_start
                data_source_type = "热钱数据"
                logger.info(f"💰 获取{data_source_type}耗时: {data_time:.2f}秒")
            
            elif any(keyword in user_input for keyword in news_keywords):
                data_start = time.time()
                market_data = await self.get_news_data()
                data_time = time.time() - data_start
                data_source_type = "新闻数据"
                logger.info(f"📰 获取{data_source_type}耗时: {data_time:.2f}秒")
            
            elif any(keyword in user_input for keyword in financial_keywords):
                # 尝试从用户输入中提取股票代码
                symbol = self.extract_stock_symbol(user_input)
                data_start = time.time()
                market_data = await self.get_financial_data(symbol=symbol)
                data_time = time.time() - data_start
                data_source_type = "财务数据"
                logger.info(f"📊 获取{data_source_type}耗时: {data_time:.2f}秒")
            
            elif any(keyword in user_input for keyword in market_keywords):
                data_start = time.time()
                market_data = await self.get_market_analysis()
                data_time = time.time() - data_start
                data_source_type = "综合分析"
                logger.info(f"🔍 获取{data_source_type}耗时: {data_time:.2f}秒")
            
            if market_data:
                # 构建包含市场数据的消息
                messages = [
                    {
                        "role": "system", 
                        "content": f"""你是专业的股票分析师。用户询问了市场相关问题，以下是基于{data_source_type}的实际市场数据：

{market_data}

请基于这些实际数据回答用户的问题。如果数据中有相关信息，请详细说明；如果没有相关数据，请告知用户数据获取情况。保持专业和客观。"""
                    },
                    {"role": "user", "content": user_input}
                ]
                
                # 选择使用的模型
                model = self.thinking_llm if use_thinking else self.llm
                
                # 调用LLM并统计时间
                llm_start = time.time()
                response = await model.a_run(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                llm_time = time.time() - llm_start
                logger.info(f"🤖 LLM响应耗时: {llm_time:.2f}秒")
                
                if response and response.content:
                    assistant_reply = response.content
                    # 保存对话历史
                    self.conversation_history.append({"role": "user", "content": user_input})
                    self.conversation_history.append({"role": "assistant", "content": assistant_reply})
                    
                    # 统计总时间
                    total_time = time.time() - start_time
                    logger.info(f"⏱️ 总响应时间: {total_time:.2f}秒")
                    
                    return assistant_reply
                else:
                    return "抱歉，无法生成基于市场数据的回复。"
            
            # 非市场问题，正常对话
            # 选择使用的模型
            model = self.thinking_llm if use_thinking else self.llm
            
            # 构建消息
            messages = [
                {
                    "role": "system", 
                    "content": "你是一位专业的金融分析师和交易顾问。你可以帮助用户分析市场数据、提供投资建议、解释金融概念等。请用中文回答，保持专业和客观。"
                }
            ]
            
            # 添加对话历史
            for msg in self.conversation_history[-10:]:  # 只保留最近10轮对话
                messages.append(msg)
            
            # 添加当前用户输入
            messages.append({"role": "user", "content": user_input})
            
            # 调用LLM并统计时间
            llm_start = time.time()
            response = await model.a_run(
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            llm_time = time.time() - llm_start
            logger.info(f"🤖 LLM响应耗时: {llm_time:.2f}秒")
            
            # 获取回复内容
            if response and response.content:
                assistant_reply = response.content
                
                # 保存对话历史
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": assistant_reply})
                
                # 统计总时间
                total_time = time.time() - start_time
                logger.info(f"⏱️ 总响应时间: {total_time:.2f}秒")
                
                return assistant_reply
            else:
                return "抱歉，我没有收到有效的回复。"
                
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ 处理失败，总耗时: {total_time:.2f}秒, 错误: {str(e)}")
            return f"发生错误: {str(e)}"
    
    async def get_market_analysis(self, date: str = None) -> str:
        """获取综合市场分析"""
        try:
            from datetime import datetime
            
            trigger_time = date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"正在获取 {trigger_time} 的综合市场分析...")
            analysis = await self.market_analyzer.get_comprehensive_analysis(trigger_time)
            
            if analysis['data_sources_count'] > 0:
                return f"📊 {analysis['trade_date']} 综合市场分析报告\n\n{analysis['comprehensive_analysis']}"
            else:
                return "⚠️ 未能获取到市场数据。"
                
        except Exception as e:
            return f"获取市场分析时发生错误: {str(e)}"
    
    async def get_market_summary(self, date: str = None) -> str:
        """获取快速市场摘要"""
        try:
            return await self.market_analyzer.get_quick_market_summary(date)
        except Exception as e:
            return f"获取市场摘要时发生错误: {str(e)}"
    
    async def get_price_data(self, date: str = None) -> str:
        """获取价格市场数据"""
        try:
            from datetime import datetime
            from data_source.price_market_akshare import PriceMarketAkshare
            
            trigger_time = date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            price_market = PriceMarketAkshare()
            df = await price_market.get_data(trigger_time)
            
            if not df.empty:
                return f"📊 价格市场数据:\n{df.iloc[0]['content'] if 'content' in df.columns else '无价格数据'}"
            else:
                return "⚠️ 未能获取价格市场数据"
        except Exception as e:
            return f"获取价格数据时发生错误: {str(e)}"
    
    async def get_hot_money_data(self, date: str = None) -> str:
        """获取热钱市场数据"""
        try:
            from datetime import datetime
            from data_source.hot_money_akshare import HotMoneyAkshare
            
            trigger_time = date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            hot_money = HotMoneyAkshare()
            df = await hot_money.get_data(trigger_time)
            
            if not df.empty:
                return f"💰 热钱市场数据:\n{df.iloc[0]['content'] if 'content' in df.columns else '无热钱数据'}"
            else:
                return "⚠️ 未能获取热钱市场数据"
        except Exception as e:
            return f"获取热钱数据时发生错误: {str(e)}"
    
    async def get_news_data(self, date: str = None) -> str:
         """获取新闻数据"""
         try:
             from datetime import datetime
             from data_source.sina_news_crawl import SinaNewsCrawl
             
             trigger_time = date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
             news_crawler = SinaNewsCrawl(start_page=1, end_page=5)
             df = await news_crawler.get_data(trigger_time)
             
             if not df.empty:
                 news_summary = []
                 for i, (_, row) in enumerate(df.head(5).iterrows(), 1):
                     title = row.get('title', '无标题')
                     content = row.get('content', '无内容')
                     if len(content) > 100:
                         content = content[:100] + "..."
                     news_summary.append(f"{i}. {title}\n   {content}")
                 
                 return f"📰 相关新闻资讯:\n" + "\n\n".join(news_summary)
             else:
                 return "⚠️ 未能获取新闻数据"
         except Exception as e:
             return f"获取新闻数据时发生错误: {str(e)}"
     
    async def get_financial_data(self, date: str = None, symbol: str = "000001") -> str:
        """获取财务数据"""
        try:
            from datetime import datetime
            from data_source.financial_statement_akshare import FinancialStatementAkshare
            
            trigger_time = date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            financial_analyzer = FinancialStatementAkshare()
            df = await financial_analyzer.get_data(trigger_time, symbol)
            
            if not df.empty:
                return f"📊 {symbol} 财务数据分析:\n{df.iloc[0]['content'] if 'content' in df.columns else '无财务数据'}"
            else:
                return f"⚠️ 未能获取 {symbol} 的财务数据"
        except Exception as e:
            return f"获取财务数据时发生错误: {str(e)}"
    
    async def get_financial_data_by_input(self, stock_input: str) -> str:
        """
        根据用户输入的股票代码或名称获取财务数据
        
        Args:
            stock_input: 用户输入的股票代码或名称
            
        Returns:
            str: 财务数据分析结果
        """
        try:
            # 提取股票代码
            symbol = self.extract_stock_symbol_with_validation(stock_input)
            
            if not symbol:
                return f"❌ 未找到股票代码或名称 '{stock_input}'，请检查输入是否正确。\n\n支持的格式：\n- 股票代码：000001、600519等\n- 股票名称：平安银行、贵州茅台等\n- 带后缀代码：000001.SH、600036.SZ等"
            
            from datetime import datetime
            from data_source.financial_statement_akshare import FinancialStatementAkshare
            
            trigger_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            financial_analyzer = FinancialStatementAkshare()
            df = await financial_analyzer.get_data(trigger_time, symbol)
            
            if not df.empty:
                return f"📊 {symbol} 财务数据分析:\n{df.iloc[0]['content'] if 'content' in df.columns else '无财务数据'}"
            else:
                return f"⚠️ 未能获取 {symbol} 的财务数据，可能是股票代码无效或数据源暂时不可用。"
        except Exception as e:
            return f"获取财务数据时发生错误: {str(e)}"
    
    async def get_stock_investment_analysis(self, stock_input: str) -> str:
        """
        根据用户输入的股票代码或名称进行完整的投资分析
        
        Args:
            stock_input: 用户输入的股票代码或名称
            
        Returns:
            str: 投资分析结果
        """
        try:
            # 提取股票代码
            symbol = self.extract_stock_symbol_with_validation(stock_input)
            
            if not symbol:
                return f"❌ 未找到股票代码或名称 '{stock_input}'，请检查输入是否正确。\n\n支持的格式：\n- 股票代码：000001、600519等\n- 股票名称：平安银行、贵州茅台等\n- 带后缀代码：000001.SH、600036.SZ等"
            
            from datetime import datetime
            
            trigger_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"🎯 正在进行 {symbol} 的完整投资分析...")
            print("📢 第一步：看涨/看跌分析师辩论...")
            
            # 进行完整的投资分析
            analysis_result = await self.analyst_manager.conduct_full_analysis(trigger_time, symbol)
            
            if not analysis_result.get('analysis_completed', False):
                return f"❌ {symbol} 投资分析失败: {analysis_result.get('error', '未知错误')}"
            
            # 格式化分析结果
            decision_result = analysis_result['decision_result']
            debate_result = analysis_result['debate_result']
            
            result_text = f"🎯 {symbol} 投资分析报告\n"
            result_text += "=" * 60 + "\n\n"
            
            # 决策结果
            result_text += f"📊 **投资决策**: {decision_result['investment_decision']}\n"
            result_text += f"🎯 **信心水平**: {decision_result['confidence_level']}\n"
            result_text += f"💰 **目标价格**: {decision_result['target_price']}\n\n"
            
            # 辩论摘要
            result_text += "📢 **分析师辩论摘要**\n"
            result_text += f"- 辩论轮次: {analysis_result['summary']['debate_rounds']}\n"
            result_text += f"- 总发言次数: {analysis_result['summary']['total_speeches']}\n\n"
            
            # 看涨方关键观点
            bull_points = decision_result['debate_summary']['bull_key_points']
            if bull_points:
                result_text += "🐂 **看涨方关键观点**:\n"
                for i, point in enumerate(bull_points, 1):
                    result_text += f"  {i}. {point}\n"
                result_text += "\n"
            
            # 看跌方关键观点
            bear_points = decision_result['debate_summary']['bear_key_points']
            if bear_points:
                result_text += "🐻 **看跌方关键观点**:\n"
                for i, point in enumerate(bear_points, 1):
                    result_text += f"  {i}. {point}\n"
                result_text += "\n"
            
            # 获胜论点
            winning_args = decision_result['debate_summary']['winning_arguments']
            if winning_args:
                result_text += "🏆 **关键获胜论点**:\n"
                for i, arg in enumerate(winning_args, 1):
                    result_text += f"  {i}. {arg}\n"
                result_text += "\n"
            
            # 详细决策分析
            result_text += "📋 **详细决策分析**:\n"
            result_text += "-" * 40 + "\n"
            result_text += decision_result['decision_analysis']
            
            return result_text
            
        except Exception as e:
            return f"❌ 进行投资分析时发生错误: {str(e)}"
    
    def extract_stock_symbol(self, user_input: str, use_default: bool = True) -> str:
        """
        从用户输入中提取股票代码（聊天模式用）
        
        Args:
            user_input: 用户输入文本
            use_default: 是否使用默认值
            
        Returns:
            str: 股票代码，如果未找到且use_default=True则返回默认值
        """
        symbol = self.extract_stock_symbol_with_validation(user_input)
        if symbol:
            return symbol
        elif use_default:
            # 如果没有找到，返回默认股票代码
            default_symbol = "000001"  # 平安银行
            logger.info(f"🔍 未找到股票代码，使用默认值: {default_symbol}")
            return default_symbol
        else:
            return None
    
    def extract_stock_symbol_with_validation(self, user_input: str) -> str:
        """
        从用户输入中提取股票代码
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            str: 股票代码，如果未找到则返回默认值
        """
        import re
        
        # 常见的股票代码模式（按优先级排序）
        patterns = [
            r'(\d{6})\.(?:SH|SZ)',  # A股带后缀 如 000001.SH, 000001.SZ (优先级最高)
            r'\b(\d{5})\b',  # 港股5位数字
            r'\b(\d{6})\b',  # A股6位数字
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, user_input)
            if matches:
                symbol = matches[0]
                # 确保是有效的股票代码格式
                if len(symbol) == 6 and symbol.isdigit():  # A股
                    logger.info(f"🔍 从用户输入中提取到A股代码: {symbol}")
                    return symbol
                elif len(symbol) == 5 and symbol.isdigit():  # 港股
                    logger.info(f"🔍 从用户输入中提取到港股代码: {symbol}")
                    return symbol
        
        # 尝试从股票名称映射到代码
        stock_name_mapping = {
            '平安银行': '000001',
            '万科A': '000002', 
            '浦发银行': '600000',
            '招商银行': '600036',
            '五粮液': '000858',
            '贵州茅台': '600519',
            '工商银行': '601398',
            '建设银行': '601939',
            '中国银行': '601988',
            '农业银行': '601288',
            '中国平安': '601318',
            '中国人寿': '601628',
            '中国石化': '600028',
            '中国石油': '601857',
            '腾讯': '00700',
            '阿里巴巴': '09988',
            '中科曙光': '603019',
            '比亚迪': '002594',
            '宁德时代': '300750',
            '海康威视': '002415',
            '美的集团': '000333',
            '格力电器': '000651',
            '海尔智家': '600690',
            '恒瑞医药': '600276',
            '药明康德': '603259',
            '迈瑞医疗': '300760',
            '东方财富': '300059',
            '中信证券': '600030',
            '海通证券': '600837',
            '华泰证券': '601688',
            '思源电气': '002028',
            '紫金矿业': '601899',
            '中国中免': '601888',
            '片仔癀': '600436',
            '长春高新': '000661',
            '立讯精密': '002475',
            '歌尔股份': '002241',
            '三一重工': '600031',
            '中联重科': '000157',
            '徐工机械': '000425',
        }
        
        for name, code in stock_name_mapping.items():
            if name in user_input:
                logger.info(f"🔍 从股票名称'{name}'映射到代码: {code}")
                return code
        
        # 如果没有找到，返回None
        logger.info(f"🔍 未找到股票代码: {user_input}")
        return None
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []
        print("对话历史已清除。")

async def main():
    """主程序"""
    print("=" * 60)
    print("🤖 投资智能体聊天助手 by LCK")
    print("=" * 60)
    print(f"模型: {cfg.llm['model_name']}")
    print(f"思考模型: {cfg.llm_thinking['model_name']}")
    print("\n可用命令:")
    print("- 直接输入问题进行对话")
    print("- 输入 'market' 获取综合市场分析")
    print("- 输入 'summary' 获取快速市场摘要")
    print("- 输入 'financial' 进入财务数据查询模式")
    print("- 输入 'analysis' 进入个股投资分析模式（看涨/看跌辩论+决策）")
    print("- 输入 'thinking' 切换到思考模式")
    print("- 输入 'normal' 切换到普通模式")
    print("- 输入 'clear' 清除对话历史")
    print("- 输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    bot = TradingChatBot()
    use_thinking = False
    
    while True:
        try:
            # 获取用户输入
            if use_thinking:
                user_input = input("\n💭 [思考模式] 你: ").strip()
            else:
                user_input = input("\n💬 [普通模式] 你: ").strip()
            
            if not user_input:
                continue
                
            # 处理特殊命令
            if user_input.lower() in ['quit', 'exit', '退出']:
                 print("👋 拜拜了您内！")
                 break
            elif user_input.lower() == 'clear':
                bot.clear_history()
                continue
            elif user_input.lower() == 'thinking':
                use_thinking = True
                print("🧠 已切换到思考模式")
                continue
            elif user_input.lower() == 'normal':
                use_thinking = False
                print("💬 已切换到普通模式")
                continue
            elif user_input.lower() == 'market':
                print("📊 正在获取综合市场分析...")
                response = await bot.get_market_analysis()
                print(f"\n🤖 助手: {response}")
                continue
            elif user_input.lower() == 'summary':
                print("📋 正在获取快速市场摘要...")
                response = await bot.get_market_summary()
                print(f"\n🤖 助手: {response}")
                continue
            elif user_input.lower() == 'financial':
                print("\n📊 进入财务数据查询模式")
                print("请输入股票代码（例如：000001）")
                print("输入 'quit' 返回主菜单")
                
                while True:
                    try:
                        stock_input = input("\n💼 [财务查询] 股票: ").strip()
                        
                        if not stock_input:
                            continue
                        
                        if stock_input.lower() == 'quit':
                            print("返回主菜单")
                            break
                        
                        print("🤔 正在查询财务数据...")
                        response = await bot.get_financial_data_by_input(stock_input)
                        print(f"\n🤖 助手: {response}")
                        
                    except KeyboardInterrupt:
                        print("\n返回主菜单")
                        break
                    except Exception as e:
                        print(f"\n❌ 发生错误: {e}")
                continue
            elif user_input.lower() == 'analysis':
                print("\n🎯 进入个股投资分析模式")
                print("将进行看涨/看跌分析师辩论，并生成投资决策")
                print("请输入股票代码（例如：000001）")
                print("输入 'quit' 返回主菜单")
                
                while True:
                    try:
                        stock_input = input("\n🎯 [投资分析] 股票: ").strip()
                        
                        if not stock_input:
                            continue
                        
                        if stock_input.lower() == 'quit':
                            print("返回主菜单")
                            break
                        
                        response = await bot.get_stock_investment_analysis(stock_input)
                        print(f"\n🤖 助手: {response}")
                        
                    except KeyboardInterrupt:
                        print("\n返回主菜单")
                        break
                    except Exception as e:
                        print(f"\n❌ 发生错误: {e}")
                continue
            
            # 显示处理中
            print("🤔 思考中...")
            
            # 与LLM对话
            response = await bot.chat(user_input, use_thinking)
            
            # 显示回复
            print(f"\n🤖 助手: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 拜拜了您内！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())
