#!/usr/bin/env python3
"""
主程序入口
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from comprehensive_analysis import ComprehensiveMarketAnalyzer
from analysts.analyst_manager import AnalystManager
from config.config import cfg

async def main():
    """主程序入口"""
    print("=" * 50)
    print("交易代理系统启动")
    print("=" * 50)
    
    # 显示配置信息
    print(f"市场类型: {cfg.market_type}")
    print(f"系统语言: {cfg.system_language}")
    print(f"LLM模型: {cfg.llm['model_name']}")
    
    # 测试综合分析系统
    print("\n测试综合市场分析系统...")
    analyzer = ComprehensiveMarketAnalyzer()
    
    try:
        # 使用当前时间作为触发时间
        from datetime import datetime
        trigger_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"使用当前时间: {trigger_time}")
        
        print("正在获取综合分析...")
        analysis = await analyzer.get_comprehensive_analysis(trigger_time)
        
        print(f"\n📊 分析结果:")
        print(f"交易日: {analysis['trade_date']}")
        print(f"数据源: {analysis['data_sources_count']}/3")
        print(f"价格数据: {'✓' if analysis['price_market_available'] else '✗'}")
        print(f"热钱数据: {'✓' if analysis['hot_money_available'] else '✗'}")
        print(f"新闻数据: {'✓' if analysis.get('news_available', False) else '✗'}")
        
        if analysis['data_sources_count'] > 0:
            print(f"\n📝 综合分析报告:")
            print("-" * 50)
            print(analysis['comprehensive_analysis'])
            print("-" * 50)
        else:
            print("⚠️ 未获取到数据")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试分析师管理系统
    print("\n" + "=" * 50)
    print("测试分析师管理系统...")
    analyst_manager = AnalystManager()
    
    try:
        # 测试个股投资分析
        test_symbol = "002028"  # 思源电气
        print(f"正在测试 {test_symbol} 的投资分析...")
        
        investment_analysis = await analyst_manager.conduct_full_analysis(trigger_time, test_symbol)
        
        if investment_analysis.get('analysis_completed', False):
            print(f"\n🎯 投资分析结果:")
            print(f"股票代码: {investment_analysis['symbol']}")
            print(f"分析完成: {investment_analysis['analysis_completed']}")
            
            decision_result = investment_analysis['decision_result']
            print(f"投资决策: {decision_result['investment_decision']}")
            print(f"信心水平: {decision_result['confidence_level']}")
            print(f"目标价格: {decision_result['target_price']}")
            
            summary = investment_analysis['summary']
            print(f"辩论轮次: {summary['debate_rounds']}")
            print(f"总发言次数: {summary['total_speeches']}")
            
            print(f"\n📋 详细决策分析:")
            print("-" * 50)
            print(decision_result['decision_analysis'])
            print("-" * 50)
        else:
            print(f"❌ 投资分析失败: {investment_analysis.get('error', '未知错误')}")
            
    except Exception as e:
        print(f"分析师管理测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
