"""
基于 akshare 的个股分析数据源
整合个股历史行情、市场热度、综合评分、机构参与度和新闻数据
"""
import pandas as pd
import asyncio
import traceback
from datetime import datetime, timedelta
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


class StockAnalysisAkshare(DataSourceBase):
    def __init__(self):
        super().__init__("stock_analysis_akshare")
        
    async def get_data(self, trigger_time: str, symbol: str) -> pd.DataFrame:
        """
        获取单个股票的全面分析数据
        
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
            logger.info(f"获取 {symbol} 在 {trade_date} 的个股分析数据")
            
            llm_summary_dict = await self.get_stock_comprehensive_analysis(symbol, trade_date, trigger_time)
            data = [{
                "title": f"{symbol} {trade_date}:个股综合分析",
                "content": llm_summary_dict["llm_summary"],
                "pub_time": trigger_time,
                "url": None,
                "symbol": symbol
            }]
            
            df = pd.DataFrame(data)
            self.save_data_cached(cache_key, df)
            return df
                
        except Exception as e:
            logger.error(f"获取个股分析数据失败: {e}")
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
    
    def get_stock_analysis_data(self, symbol: str, trigger_time: str = None) -> Dict[str, Any]:
        """
        获取股票全面分析数据
        
        Args:
            symbol: 股票代码 (6位数字)
            trigger_time: 触发时间，用于计算交易日
            
        Returns:
            Dict: 包含历史行情、热度、评分、机构参与度和新闻的数据
        """
        try:
            logger.info(f"🔍 开始获取{symbol}的个股分析数据")
            
            # 转换股票代码格式
            formatted_symbol = self._convert_symbol_format(symbol)
            logger.debug(f"📊 股票代码格式转换: {symbol} -> {formatted_symbol}")
            
            analysis_data = {}
            
            # 1. 获取个股主营介绍
            try:
                logger.debug(f"🏢 尝试获取{symbol}主营介绍...")
                business_intro_data = akshare_cached.run(
                    func_name="stock_zyjs_ths",
                    func_kwargs={"symbol": symbol},
                    verbose=False
                )
                if business_intro_data is not None and not business_intro_data.empty:
                    analysis_data['business_introduction'] = business_intro_data
                    logger.info(f"✅ 成功获取{symbol}主营介绍: {len(business_intro_data)}条记录")
                    logger.debug(f"主营介绍列名: {list(business_intro_data.columns)}")
                else:
                    logger.warning(f"⚠️ {symbol}主营介绍为空")
            except Exception as e:
                logger.warning(f"❌ 获取{symbol}主营介绍失败: {e}")
            
            # 2. 获取个股历史行情（近三年）
            try:
                logger.debug(f"📈 尝试获取{symbol}历史行情...")
                # 使用交易日作为结束日期，确保获取最新数据
                from utils.date_utils import get_smart_trading_date
                end_date = get_smart_trading_date(trigger_time)
                start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')
                
                hist_data = akshare_cached.run(
                    func_name="stock_zh_a_hist",
                    func_kwargs={
                        "symbol": symbol,
                        "period": "daily",
                        "start_date": start_date,
                        "end_date": end_date,
                        "adjust": "qfq"
                    },
                    verbose=False
                )
                if hist_data is not None and not hist_data.empty:
                    analysis_data['historical_data'] = hist_data
                    logger.info(f"✅ 成功获取{symbol}历史行情: {len(hist_data)}条记录")
                    logger.debug(f"历史行情列名: {list(hist_data.columns)}")
                else:
                    logger.warning(f"⚠️ {symbol}历史行情为空")
            except Exception as e:
                logger.warning(f"❌ 获取{symbol}历史行情失败: {e}")
            
            # 3. 获取个股用户关注指数
            try:
                logger.debug(f"🔥 尝试获取{symbol}用户关注指数...")
                market_heat = akshare_cached.run(
                    func_name="stock_comment_detail_scrd_focus_em",
                    func_kwargs={"symbol": symbol},
                    verbose=False
                )
                if market_heat is not None and not market_heat.empty:
                    analysis_data['market_heat'] = market_heat
                    logger.info(f"✅ 成功获取{symbol}用户关注指数: {len(market_heat)}条记录")
                    logger.debug(f"用户关注指数列名: {list(market_heat.columns)}")
                else:
                    logger.warning(f"⚠️ {symbol}用户关注指数为空")
            except Exception as e:
                logger.warning(f"❌ 获取{symbol}用户关注指数失败: {e}")
            
            # 4. 获取个股机构参与度
            try:
                logger.debug(f"🏛️ 尝试获取{symbol}机构参与度...")
                institution_data = akshare_cached.run(
                    func_name="stock_comment_detail_zlkp_jgcyd_em",
                    func_kwargs={"symbol": symbol},
                    verbose=False
                )
                if institution_data is not None and not institution_data.empty:
                    analysis_data['institution_participation'] = institution_data
                    logger.info(f"✅ 成功获取{symbol}机构参与度: {len(institution_data)}条记录")
                    logger.debug(f"机构参与度列名: {list(institution_data.columns)}")
                else:
                    logger.warning(f"⚠️ {symbol}机构参与度为空")
            except Exception as e:
                logger.warning(f"❌ 获取{symbol}机构参与度失败: {e}")
            
            # 5. 获取个股市场参与度
            try:
                logger.debug(f"📊 尝试获取{symbol}市场参与度...")
                market_desire_data = akshare_cached.run(
                    func_name="stock_comment_detail_scrd_desire_daily_em",
                    func_kwargs={"symbol": symbol},
                    verbose=False
                )
                if market_desire_data is not None and not market_desire_data.empty:
                    analysis_data['market_desire'] = market_desire_data
                    logger.info(f"✅ 成功获取{symbol}市场参与度: {len(market_desire_data)}条记录")
                    logger.debug(f"市场参与度列名: {list(market_desire_data.columns)}")
                else:
                    logger.warning(f"⚠️ {symbol}市场参与度为空")
            except Exception as e:
                logger.warning(f"❌ 获取{symbol}市场参与度失败: {e}")
            
            # 6. 获取个股综合评价
            try:
                logger.debug(f"⭐ 尝试获取{symbol}综合评价...")
                comprehensive_rating_data = akshare_cached.run(
                    func_name="stock_comment_detail_zhpj_lspf_em",
                    func_kwargs={"symbol": symbol},
                    verbose=False
                )
                if comprehensive_rating_data is not None and not comprehensive_rating_data.empty:
                    analysis_data['comprehensive_rating'] = comprehensive_rating_data
                    logger.info(f"✅ 成功获取{symbol}综合评价: {len(comprehensive_rating_data)}条记录")
                    logger.debug(f"综合评价列名: {list(comprehensive_rating_data.columns)}")
                else:
                    logger.warning(f"⚠️ {symbol}综合评价为空")
            except Exception as e:
                logger.warning(f"❌ 获取{symbol}综合评价失败: {e}")
            
            # 7. 获取个股估值数据
            try:
                logger.debug(f"💰 尝试获取{symbol}个股估值...")
                valuation_data = akshare_cached.run(
                    func_name="stock_value_em",
                    func_kwargs={"symbol": symbol},
                    verbose=False
                )
                if valuation_data is not None and not valuation_data.empty:
                    analysis_data['stock_valuation'] = valuation_data
                    logger.info(f"✅ 成功获取{symbol}个股估值: {len(valuation_data)}条记录")
                    logger.debug(f"个股估值列名: {list(valuation_data.columns)}")
                else:
                    logger.warning(f"⚠️ {symbol}个股估值为空")
            except Exception as e:
                logger.warning(f"❌ 获取{symbol}个股估值失败: {e}")
            
            # 8. 获取个股新闻
            try:
                logger.debug(f"📰 尝试获取{symbol}个股新闻...")
                news_data = akshare_cached.run(
                    func_name="stock_news_em",
                    func_kwargs={"symbol": symbol},
                    verbose=False
                )
                if news_data is not None and not news_data.empty:
                    analysis_data['stock_news'] = news_data
                    logger.info(f"✅ 成功获取{symbol}个股新闻: {len(news_data)}条记录")
                    logger.debug(f"个股新闻列名: {list(news_data.columns)}")
                else:
                    logger.warning(f"⚠️ {symbol}个股新闻为空")
            except Exception as e:
                logger.warning(f"❌ 获取{symbol}个股新闻失败: {e}")
            
            # 记录最终结果
            if analysis_data:
                logger.info(f"✅ 个股分析数据获取完成: {symbol} (格式: {formatted_symbol}), 包含{len(analysis_data)}个数据集")
                for key, value in analysis_data.items():
                    if hasattr(value, '__len__'):
                        logger.info(f"  - {key}: {len(value)}条记录")
            else:
                logger.warning(f"⚠️ 未能获取{symbol}的任何个股分析数据")
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"❌ 获取{symbol}个股分析数据失败: {e}")
            return {}
    
    def format_stock_analysis_data(self, analysis_data: Dict[str, Any], symbol: str) -> str:
        """
        格式化个股分析数据为文本
        """
        if not analysis_data:
            return f"{symbol} 无个股分析数据"
        
        formatted_text = f"📊 {symbol} 个股综合分析数据:\n\n"
        
        # 个股主营介绍数据
        if 'business_introduction' in analysis_data:
            business_data = analysis_data['business_introduction']
            if not business_data.empty:
                formatted_text += "## 个股主营介绍:\n\n"
                
                # 显示主营介绍信息
                for _, row in business_data.iterrows():
                    try:
                        stock_code = row.get('股票代码', 'N/A')
                        main_business = row.get('主营业务', 'N/A')
                        product_type = row.get('产品类型', 'N/A')
                        product_name = row.get('产品名称', 'N/A')
                        business_scope = row.get('经营范围', 'N/A')
                        
                        formatted_text += f"股票代码: {stock_code}\n"
                        formatted_text += f"主营业务: {main_business}\n"
                        formatted_text += f"产品类型: {product_type}\n"
                        formatted_text += f"产品名称: {product_name}\n"
                        formatted_text += f"经营范围: {business_scope}\n\n"
                    except Exception as e:
                        logger.warning(f"格式化主营介绍数据行失败: {e}")
                        continue
        
        # 历史行情数据
        if 'historical_data' in analysis_data:
            hist_data = analysis_data['historical_data']
            if not hist_data.empty:
                formatted_text += "## 历史行情数据:\n\n"
                
                # 显示最新几个交易日的数据
                recent_data = hist_data.tail(20)
                
                # 计算关键指标
                if len(hist_data) > 1:
                    latest_price = hist_data.iloc[0]['收盘'] if '收盘' in hist_data.columns else 'N/A'
                    prev_price = hist_data.iloc[1]['收盘'] if '收盘' in hist_data.columns else 'N/A'
                    
                    if latest_price != 'N/A' and prev_price != 'N/A':
                        price_change = latest_price - prev_price
                        price_change_pct = (price_change / prev_price) * 100
                        formatted_text += f"最新收盘价: {latest_price:.2f}元\n"
                        formatted_text += f"涨跌幅: {price_change:+.2f}元 ({price_change_pct:+.2f}%)\n\n"
                
                # 显示最近交易日数据
                for _, row in recent_data.iterrows():
                    try:
                        date = row.get('日期', 'N/A')
                        close_price = row.get('收盘', 'N/A')
                        open_price = row.get('开盘', 'N/A')
                        high_price = row.get('最高', 'N/A')
                        low_price = row.get('最低', 'N/A')
                        volume = row.get('成交量', 'N/A')
                        amount = row.get('成交额', 'N/A')
                        change_pct = row.get('涨跌幅', 'N/A')
                        turnover_rate = row.get('换手率', 'N/A')
                        
                        formatted_text += f"日期: {date}\n"
                        formatted_text += f"  开盘价: {open_price:.2f}元\n"
                        formatted_text += f"  收盘价: {close_price:.2f}元\n"
                        formatted_text += f"  最高价: {high_price:.2f}元\n"
                        formatted_text += f"  最低价: {low_price:.2f}元\n"
                        formatted_text += f"  成交量: {volume:.0f}\n"
                        formatted_text += f"  成交额: {amount:.2f}万元\n"
                        formatted_text += f"  涨跌幅: {change_pct:+.2f}%\n"
                        formatted_text += f"  换手率: {turnover_rate:.2f}%\n\n"
                    except Exception as e:
                        logger.warning(f"格式化历史数据行失败: {e}")
                        continue
        
        # 用户关注指数数据
        if 'market_heat' in analysis_data:
            heat_data = analysis_data['market_heat']
            if not heat_data.empty:
                formatted_text += "## 用户关注指数分析:\n\n"
                
                # 显示用户关注指数（限制显示最新10条）
                for _, row in heat_data.tail(10).iterrows():
                    try:
                        date = row.get('交易日', 'N/A')
                        focus_index = row.get('用户关注指数', 'N/A')
                        
                        formatted_text += f"交易日: {date}\n"
                        formatted_text += f"  用户关注指数: {focus_index}\n\n"
                    except Exception as e:
                        logger.warning(f"格式化用户关注指数数据行失败: {e}")
                        continue
        
        # 机构参与度数据
        if 'institution_participation' in analysis_data:
            inst_data = analysis_data['institution_participation']
            if not inst_data.empty:
                formatted_text += "## 机构参与度分析:\n\n"
                
                # 显示机构参与度指标（限制显示最新10条）
                for _, row in inst_data.tail(10).iterrows():
                    try:
                        date = row.get('交易日', 'N/A')
                        institution_participation = row.get('机构参与度', 'N/A')
                        
                        formatted_text += f"交易日: {date}\n"
                        formatted_text += f"  机构参与度: {institution_participation}\n\n"
                    except Exception as e:
                        logger.warning(f"格式化机构参与度数据行失败: {e}")
                        continue
        
        # 市场参与度数据
        if 'market_desire' in analysis_data:
            desire_data = analysis_data['market_desire']
            if not desire_data.empty:
                formatted_text += "## 市场参与度分析:\n\n"
                
                # 显示市场参与度指标（限制显示最新10条）
                for _, row in desire_data.tail(10).iterrows():
                    try:
                        date = row.get('交易日', 'N/A')
                        daily_desire = row.get('当日意愿上升', 'N/A')
                        avg_desire_change = row.get('5日平均参与意愿变化', 'N/A')
                        
                        formatted_text += f"交易日: {date}\n"
                        formatted_text += f"  当日意愿上升: {daily_desire}\n"
                        formatted_text += f"  5日平均参与意愿变化: {avg_desire_change}\n\n"
                    except Exception as e:
                        logger.warning(f"格式化市场参与度数据行失败: {e}")
                        continue
        
        # 综合评价数据
        if 'comprehensive_rating' in analysis_data:
            rating_data = analysis_data['comprehensive_rating']
            if not rating_data.empty:
                formatted_text += "## 综合评价分析:\n\n"
                
                # 显示综合评价指标（限制显示最新10条）
                for _, row in rating_data.tail(10).iterrows():
                    try:
                        date = row.get('交易日', 'N/A')
                        rating = row.get('评分', 'N/A')
                        
                        formatted_text += f"交易日: {date}\n"
                        formatted_text += f"  评分: {rating}\n\n"
                    except Exception as e:
                        logger.warning(f"格式化综合评价数据行失败: {e}")
                        continue
        
        # 个股估值数据
        if 'stock_valuation' in analysis_data:
            valuation_data = analysis_data['stock_valuation']
            if not valuation_data.empty:
                formatted_text += "## 个股估值分析:\n\n"
                
                # 显示估值指标（限制显示最新10条）
                for _, row in valuation_data.tail(10).iterrows():
                    try:
                        data_date = row.get('数据日期', 'N/A')
                        close_price = row.get('当日收盘价', 'N/A')
                        change_pct = row.get('当日涨跌幅', 'N/A')
                        total_market_value = row.get('总市值', 'N/A')
                        circulating_market_value = row.get('流通市值', 'N/A')
                        total_shares = row.get('总股本', 'N/A')
                        circulating_shares = row.get('流通股本', 'N/A')
                        pe_ttm = row.get('PE(TTM)', 'N/A')
                        pe_static = row.get('PE(静)', 'N/A')
                        pb_ratio = row.get('市净率', 'N/A')
                        peg_ratio = row.get('PEG值', 'N/A')
                        pcf_ratio = row.get('市现率', 'N/A')
                        ps_ratio = row.get('市销率', 'N/A')
                        
                        formatted_text += f"数据日期: {data_date}\n"
                        formatted_text += f"  当日收盘价: {close_price}元\n"
                        formatted_text += f"  当日涨跌幅: {change_pct}%\n"
                        formatted_text += f"  总市值: {total_market_value}万元\n"
                        formatted_text += f"  流通市值: {circulating_market_value}万元\n"
                        formatted_text += f"  总股本: {total_shares}万股\n"
                        formatted_text += f"  流通股本: {circulating_shares}万股\n"
                        formatted_text += f"  PE(TTM): {pe_ttm}\n"
                        formatted_text += f"  PE(静): {pe_static}\n"
                        formatted_text += f"  市净率: {pb_ratio}\n"
                        formatted_text += f"  PEG值: {peg_ratio}\n"
                        formatted_text += f"  市现率: {pcf_ratio}\n"
                        formatted_text += f"  市销率: {ps_ratio}\n\n"
                    except Exception as e:
                        logger.warning(f"格式化估值数据行失败: {e}")
                        continue
        
        # 个股新闻数据
        if 'stock_news' in analysis_data:
            news_data = analysis_data['stock_news']
            if not news_data.empty:
                formatted_text += "## 个股新闻资讯:\n\n"
                
                # 显示最新新闻
                recent_news = news_data.head(5)
                for _, row in recent_news.iterrows():
                    try:
                        keyword = row.get('关键词', 'N/A')
                        title = row.get('新闻标题', 'N/A')
                        content = row.get('新闻内容', 'N/A')
                        pub_time = row.get('发布时间', 'N/A')
                        source = row.get('文章来源', 'N/A')
                        url = row.get('新闻链接', 'N/A')
                        
                        formatted_text += f"关键词: {keyword}\n"
                        formatted_text += f"标题: {title}\n"
                        formatted_text += f"时间: {pub_time}\n"
                        formatted_text += f"来源: {source}\n"
                        if content != 'N/A' and len(content) > 150:
                            formatted_text += f"内容: {content[:150]}...\n"
                        elif content != 'N/A':
                            formatted_text += f"内容: {content}\n"
                        formatted_text += "\n"
                    except Exception as e:
                        logger.warning(f"格式化新闻数据行失败: {e}")
                        continue
        
        return formatted_text
    
    async def get_stock_comprehensive_analysis(self, symbol: str, trade_date: str, trigger_time: str = None) -> dict:
        """
        获取单个股票的数据汇总
        
        Args:
            symbol: 股票代码
            trade_date: 交易日
            trigger_time: 触发时间
        """
        try:
            logger.info(f"获取 {symbol} 在 {trade_date} 的数据汇总")
            
            # 获取个股分析数据
            analysis_data = self.get_stock_analysis_data(symbol, trigger_time)
            
            if not analysis_data:
                return {
                    'trade_date': trade_date,
                    'symbol': symbol,
                    'raw_data': "无个股分析数据",
                    'llm_summary': f"{symbol} 在 {trade_date} 无可用个股分析数据",
                    'data_count': 0
                }
            
            # 格式化个股数据
            formatted_data = self.format_stock_analysis_data(analysis_data, symbol)
            
            # 构建LLM总结提示词
            prompt = f"""
请总结以下{symbol}股票的全面数据，并给出结构化的信息汇总报告（1500字符以内）：

⚠️ **时间提醒**：
- 分析时间：{trigger_time}
- 交易日：{trade_date}
- 请确保在汇总中明确指出数据的时间范围，避免引用过时的价格信息

## 个股数据汇总
{formatted_data}

## 总结要求
请基于提供的个股数据，进行以下维度的信息总结：

1. **历史行情总结**：
   - 总结价格走势和成交量变化
   - 归纳关键价格点位和波动特征
   - 整理重要的交易数据

2. **用户关注指数总结**：
   - 总结用户关注指数的变化趋势
   - 归纳投资者关注度变化特征

3. **机构参与度总结**：
   - 总结机构参与度指标变化
   - 归纳机构资金参与情况

4. **市场参与度总结**：
   - 总结当日意愿上升变化
   - 归纳5日平均参与意愿变化趋势

5. **综合评价总结**：
   - 总结评分变化情况
   - 归纳评级变化趋势

6. **个股估值总结**：
   - 总结市值和股本情况
   - 归纳估值指标变化（PE、PB、PEG等）
   - 整理估值水平分析

7. **新闻资讯总结**：
   - 总结最新相关新闻要点
   - 归纳重要事件和时间节点
   - 整理消息面关键信息

## 输出要求
- 专注于数据总结和信息归纳，不进行投资分析
- 保持客观中立，如实反映数据内容
- 结构化整理信息，便于后续分析使用
- 需要包含个股主营介绍
- 控制在1500字符以内
"""
            
            messages = [
                {
                    "role": "system", 
                    "content": "你是一位专业的数据整理专家，专长于股票数据汇总和信息归纳。请基于实际数据生成结构化的信息汇总报告，不进行投资分析。"
                },
                {"role": "user", "content": prompt}
            ]
            
            response = await GLOBAL_LLM.a_run(
                messages=messages,
                thinking=False,
                temperature=0.3,
                max_tokens=2000
            )
            
            if response and response.content:
                llm_summary = response.content
            else:
                logger.error(f"LLM数据汇总未返回内容")
                llm_summary = f"{symbol} 数据汇总失败"
            
            return {
                'trade_date': trade_date,
                'symbol': symbol,
                'raw_data': formatted_data,
                'llm_summary': llm_summary,
                'data_count': len(analysis_data)
            }
                
        except Exception as e:
            traceback.print_exc()
            logger.error(f"获取{symbol}数据汇总失败: {e}")
            return {
                'trade_date': trade_date,
                'symbol': symbol,
                'raw_data': "数据获取失败",
                'llm_summary': f"{symbol} 数据汇总失败: {str(e)}",
                'data_count': 0
            }


if __name__ == "__main__":
    # 测试个股分析数据源
    stock_analyzer = StockAnalysisAkshare()
    df = asyncio.run(stock_analyzer.get_data("2024-08-19 09:00:00", "000001"))
    print(df.content.values[0] if not df.empty else "无数据")
