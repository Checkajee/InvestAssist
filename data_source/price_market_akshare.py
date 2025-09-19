"""
基于 akshare 的价格市场数据源
整合K线数据、板块资金流向等，生成综合宏观市场分析
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
from matplotlib.patches import Rectangle

class PriceMarketAkshare(DataSourceBase):
    def __init__(self):
        super().__init__("price_market_akshare")
        
    async def get_data(self, trigger_time: str) -> pd.DataFrame:
        try:
            df = self.get_data_cached(trigger_time)
            if df is not None:
                return df
            
            trade_date = get_smart_trading_date(trigger_time)     
            logger.info(f"获取 {trade_date} 的价格市场数据")

            llm_summary_dict = await self.get_llm_summary(trade_date)
            data = [{
                "title": f"{trade_date}:市场宏观数据汇总",
                "content": llm_summary_dict["llm_summary"],
                "pub_time": trigger_time,
                "url": None
            }]
            df = pd.DataFrame(data)
            self.save_data_cached(trigger_time, df)
            return df
                
        except Exception as e:
            logger.error(f"获取价格市场数据失败: {e}")
            return pd.DataFrame()
    
    def get_kline_data(self, trade_date: str) -> dict:
        """
        获取三大指数的K线数据
        """
        try:
            indices = {
                "000001.SH": {"symbol": "sh000001", "name": "上证指数"},
                "399006.SZ": {"symbol": "sz399006", "name": "创业板指"},
                "000688.SH": {"symbol": "sh000688", "name": "科创50"}
            }
            
            kline_data = {}
            
            for stock_code, info in indices.items():
                try:
                    # 获取指数历史数据
                    df = akshare_cached.run(
                        func_name="stock_zh_index_daily",
                        func_kwargs={"symbol": info["symbol"]},
                        verbose=False
                    )
                    
                    if df.empty:
                        logger.warning(f"{info['name']} 数据为空")
                        continue
                    
                    # 转换日期格式并筛选最近90天的数据
                    df['date'] = pd.to_datetime(df['date'])
                    target_date = datetime.strptime(trade_date, '%Y%m%d')
                    
                    filtered_df = df[df['date'] <= target_date].tail(90)
                    
                    if filtered_df.empty:
                        logger.warning(f"{info['name']} 无{trade_date}之前的数据")
                        continue
                    
                    # 转换为所需格式
                    data_list = []
                    for _, row in filtered_df.iterrows():
                        data_list.append({
                            'trade_date': row['date'].strftime('%Y%m%d'),
                            'open_price': float(row['open']),
                            'high_price': float(row['high']),
                            'low_price': float(row['low']),
                            'close_price': float(row['close']),
                            'trade_lots': int(row['volume'])
                        })
                    
                    kline_data[stock_code] = {
                        'name': info['name'],
                        'data': data_list
                    }
                    
                    logger.info(f"获取 {info['name']} K线数据成功，{len(data_list)} 条记录")
                    
                except Exception as e:
                    logger.error(f"获取 {info['name']} K线数据失败: {e}")
                    continue
            
            return kline_data
            
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return {}
    
    def get_current_day_data(self, trade_date: str) -> dict:
        """
        获取三大指数当日收盘数据
        """
        try:
            indices = {
                "000001.SH": {"symbol": "sh000001", "name": "上证指数"},
                "399006.SZ": {"symbol": "sz399006", "name": "创业板指"},
                "000688.SH": {"symbol": "sh000688", "name": "科创50"}
            }
            
            current_day_data = {}
            
            for stock_code, info in indices.items():
                try:
                    # 获取指数历史数据
                    df = akshare_cached.run(
                        func_name="stock_zh_index_daily",
                        func_kwargs={"symbol": info["symbol"]},
                        verbose=False
                    )
                    
                    if df.empty:
                        logger.warning(f"{info['name']} 数据为空")
                        continue
                    
                    # 转换日期格式并查找指定日期的数据
                    df['date'] = pd.to_datetime(df['date'])
                    target_date = datetime.strptime(trade_date, '%Y%m%d')
                    
                    # 查找指定日期的数据
                    target_row = df[df['date'] == target_date]
                    
                    if target_row.empty:
                        # 如果没有当日数据，取最近的一条数据
                        target_row = df[df['date'] <= target_date].tail(1)
                        if target_row.empty:
                            logger.warning(f"{info['name']} 无{trade_date}的数据")
                            continue
                    
                    row = target_row.iloc[0]
                    
                    # 计算涨跌幅（需要前一天的数据）
                    prev_row = df[df['date'] < row['date']].tail(1)
                    if not prev_row.empty:
                        prev_close = float(prev_row.iloc[0]['close'])
                        price_change = float(row['close']) - prev_close
                        price_change_rate = price_change / prev_close
                    else:
                        price_change = 0.0
                        price_change_rate = 0.0
                    
                    current_day_data[stock_code] = {
                        'name': info['name'],
                        'open_price': float(row['open']),
                        'high_price': float(row['high']),
                        'low_price': float(row['low']),
                        'close_price': float(row['close']),
                        'price_change': price_change,
                        'price_change_rate': price_change_rate,
                        'trade_amount': float(row['volume']) * float(row['close']),  # 估算成交额
                        'trade_lots': int(row['volume'])
                    }
                    
                    logger.info(f"获取 {info['name']} 当日数据成功")
                    
                except Exception as e:
                    logger.error(f"获取 {info['name']} 当日数据失败: {e}")
                    continue
            
            return current_day_data
            
        except Exception as e:
            logger.error(f"获取当日数据失败: {e}")
            return {}
    
    def get_sector_summary(self, trade_date: str) -> str:
        """
        获取板块资金流向摘要
        """
        try:
            # 获取板块资金流向数据
            df = akshare_cached.run(
                func_name="stock_board_industry_name_em",
                func_kwargs={},
                verbose=False
            )
            
            if df.empty:
                return "无板块资金流向数据"
            
            summary_lines = [f"{trade_date} 板块资金流向情况（东方财富数据）：\n"]
            
            # 取前10个板块
            top_sectors = df.head(10)
            
            for _, row in top_sectors.iterrows():
                try:
                    sector_name = row['板块名称']
                    latest_price = row['最新价']
                    change_amount = row['涨跌额']
                    change_rate = row['涨跌幅']
                    market_cap = row['总市值'] / 100000000  # 转换为亿元
                    turnover_rate = row['换手率']
                    up_count = row['上涨家数']
                    down_count = row['下跌家数']
                    leading_stock = row['领涨股票']
                    leading_change = row['领涨股票-涨跌幅']
                    
                    change_sign = "+" if change_amount >= 0 else ""
                    rate_sign = "+" if change_rate >= 0 else ""
                    
                    summary_lines.append(
                        f"**{sector_name}**: 最新价 {latest_price:.2f}, "
                        f"涨跌 {change_sign}{change_amount:.2f} ({rate_sign}{change_rate:.2f}%), "
                        f"总市值 {market_cap:.0f}亿, 换手率 {turnover_rate:.2f}%, "
                        f"上涨 {up_count} 下跌 {down_count}, 领涨股 {leading_stock} ({leading_change:+.2f}%)"
                    )
                except Exception as e:
                    logger.warning(f"处理板块数据行失败: {e}")
                    continue
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"获取板块资金流向失败: {e}")
            return f"获取板块资金流向失败: {str(e)}"
    
    def get_market_indicators(self, trade_date: str) -> dict:
        """
        获取关键市场指标数据（近90天历史数据）
        """
        indicators = {}
        
        try:
            # 1. 沪深300期权波动率(qvix) - 市场恐慌程度
            try:
                qvix_df = akshare_cached.run(
                    func_name="index_option_300etf_qvix",
                    func_kwargs={},
                    verbose=False
                )
                
                if not qvix_df.empty:
                    # 转换日期格式并筛选最近90天的数据
                    qvix_df = qvix_df.copy()
                    qvix_df['date'] = pd.to_datetime(qvix_df['date'])
                    target_date = datetime.strptime(trade_date, '%Y%m%d')
                    
                    # 筛选最近90天的数据
                    filtered_df = qvix_df[qvix_df['date'] <= target_date].tail(90)
                    
                    if not filtered_df.empty:
                        # 获取最新数据
                        latest_row = filtered_df.iloc[-1]
                        # 计算统计信息
                        recent_data = filtered_df['close'].tail(30)  # 最近30天数据
                        
                        indicators['qvix'] = {
                            'latest': {
                                'date': latest_row['date'].strftime('%Y-%m-%d'),
                                'open': float(latest_row['open']),
                                'high': float(latest_row['high']),
                                'low': float(latest_row['low']),
                                'close': float(latest_row['close'])
                            },
                            'history': filtered_df.to_dict('records'),
                            'stats': {
                                'current': float(latest_row['close']),
                                'avg_30d': float(recent_data.mean()),
                                'max_30d': float(recent_data.max()),
                                'min_30d': float(recent_data.min()),
                                'volatility_30d': float(recent_data.std())
                            }
                        }
                        logger.info(f"获取QVIX历史数据成功: 当前{indicators['qvix']['latest']['close']:.2f}, 30日均值{indicators['qvix']['stats']['avg_30d']:.2f}")
            except Exception as e:
                logger.warning(f"获取QVIX数据失败: {e}")
            
            # 2. 巴菲特指标 - 市场估值水平
            try:
                buffett_df = akshare_cached.run(
                    func_name="stock_buffett_index_lg",
                    func_kwargs={},
                    verbose=False
                )
                
                if not buffett_df.empty:
                    # 转换日期格式并筛选最近90天的数据
                    buffett_df = buffett_df.copy()
                    buffett_df['日期'] = pd.to_datetime(buffett_df['日期'])
                    target_date = datetime.strptime(trade_date, '%Y%m%d')
                    
                    # 筛选最近90天的数据
                    filtered_df = buffett_df[buffett_df['日期'] <= target_date].tail(90)
                    
                    if not filtered_df.empty:
                        # 获取最新数据
                        latest_row = filtered_df.iloc[-1]
                        # 计算统计信息
                        recent_data = filtered_df['收盘价'].tail(30)  # 最近30天数据
                        
                        # 计算巴菲特指标（总市值/GDP * 100）
                        buffett_ratio = float(latest_row['总市值']) / float(latest_row['GDP']) * 100
                        recent_ratios = filtered_df['总市值'].tail(30) / filtered_df['GDP'].tail(30) * 100
                        
                        indicators['buffett'] = {
                            'latest': {
                                'date': latest_row['日期'].strftime('%Y-%m-%d'),
                                'close_price': buffett_ratio,  # 使用计算出的正确比率
                                'market_cap': float(latest_row['总市值']),
                                'gdp': float(latest_row['GDP']),
                                'ten_year_percentile': float(latest_row['近十年分位数']),
                                'total_percentile': float(latest_row['总历史分位数'])
                            },
                            'history': filtered_df.to_dict('records'),
                            'stats': {
                                'current': buffett_ratio,
                                'avg_30d': float(recent_ratios.mean()),
                                'max_30d': float(recent_ratios.max()),
                                'min_30d': float(recent_ratios.min()),
                                'trend_30d': float(recent_ratios.iloc[-1] - recent_ratios.iloc[0])
                            }
                        }
                        logger.info(f"获取巴菲特指标历史数据成功: 当前{indicators['buffett']['latest']['close_price']:.2f}%, 30日均值{indicators['buffett']['stats']['avg_30d']:.2f}%")
            except Exception as e:
                logger.warning(f"获取巴菲特指标失败: {e}")
            
            # 3. 股债利差 - 市场风险偏好
            try:
                ebs_df = akshare_cached.run(
                    func_name="stock_ebs_lg",
                    func_kwargs={},
                    verbose=False
                )
                
                if not ebs_df.empty:
                    # 转换日期格式并筛选最近90天的数据
                    ebs_df = ebs_df.copy()
                    ebs_df['日期'] = pd.to_datetime(ebs_df['日期'])
                    target_date = datetime.strptime(trade_date, '%Y%m%d')
                    
                    # 筛选最近90天的数据
                    filtered_df = ebs_df[ebs_df['日期'] <= target_date].tail(90)
                    
                    if not filtered_df.empty:
                        # 获取最新数据
                        latest_row = filtered_df.iloc[-1]
                        # 计算统计信息
                        recent_spread = filtered_df['股债利差'].tail(30)  # 最近30天数据
                        
                        # 将股债利差转换为百分比
                        spread_percent = float(latest_row['股债利差']) * 100
                        spread_ma_percent = float(latest_row['股债利差均线']) * 100
                        recent_spread_percent = recent_spread * 100
                        
                        indicators['ebs'] = {
                            'latest': {
                                'date': latest_row['日期'].strftime('%Y-%m-%d'),
                                'hs300_index': float(latest_row['沪深300指数']),
                                'stock_bond_spread': spread_percent,
                                'spread_ma': spread_ma_percent
                            },
                            'history': filtered_df.to_dict('records'),
                            'stats': {
                                'current_spread': spread_percent,
                                'avg_spread_30d': float(recent_spread_percent.mean()),
                                'max_spread_30d': float(recent_spread_percent.max()),
                                'min_spread_30d': float(recent_spread_percent.min()),
                                'trend_30d': float(recent_spread_percent.iloc[-1] - recent_spread_percent.iloc[0])
                            }
                        }
                        logger.info(f"获取股债利差历史数据成功: 当前{indicators['ebs']['latest']['stock_bond_spread']:.2f}%, 30日均值{indicators['ebs']['stats']['avg_spread_30d']:.2f}%")
            except Exception as e:
                logger.warning(f"获取股债利差失败: {e}")
            
            # 4. 大盘拥挤度 - 市场微观结构
            try:
                congestion_df = akshare_cached.run(
                    func_name="stock_a_congestion_lg",
                    func_kwargs={},
                    verbose=False
                )
                
                if not congestion_df.empty:
                    # 转换日期格式并筛选最近90天的数据
                    congestion_df = congestion_df.copy()
                    congestion_df['date'] = pd.to_datetime(congestion_df['date'])
                    target_date = datetime.strptime(trade_date, '%Y%m%d')
                    
                    # 筛选最近90天的数据
                    filtered_df = congestion_df[congestion_df['date'] <= target_date].tail(90)
                    
                    if not filtered_df.empty:
                        # 获取最新数据
                        latest_row = filtered_df.iloc[-1]
                        # 计算统计信息
                        recent_congestion = filtered_df['congestion'].tail(30)  # 最近30天数据
                        
                        # 将拥挤度转换为百分比
                        congestion_percent = float(latest_row['congestion']) * 100
                        recent_congestion_percent = recent_congestion * 100
                        
                        indicators['congestion'] = {
                            'latest': {
                                'date': latest_row['date'].strftime('%Y-%m-%d'),
                                'close': float(latest_row['close']),
                                'congestion': congestion_percent
                            },
                            'history': filtered_df.to_dict('records'),
                            'stats': {
                                'current': congestion_percent,
                                'avg_30d': float(recent_congestion_percent.mean()),
                                'max_30d': float(recent_congestion_percent.max()),
                                'min_30d': float(recent_congestion_percent.min()),
                                'trend_30d': float(recent_congestion_percent.iloc[-1] - recent_congestion_percent.iloc[0])
                            }
                        }
                        logger.info(f"获取大盘拥挤度历史数据成功: 当前{indicators['congestion']['latest']['congestion']:.2f}%, 30日均值{indicators['congestion']['stats']['avg_30d']:.2f}%")
            except Exception as e:
                logger.warning(f"获取大盘拥挤度失败: {e}")
            
            return indicators
            
        except Exception as e:
            logger.error(f"获取市场指标失败: {e}")
            return {}
    
    def format_market_indicators(self, indicators: dict, trade_date: str) -> str:
        """
        格式化市场指标数据为文本（包含历史趋势分析）
        """
        if not indicators:
            return f"{trade_date} 无市场指标数据"
        
        summary_lines = [f"{trade_date} 关键市场指标（含90天历史趋势）：\n"]
        
        # QVIX - 市场恐慌程度
        if 'qvix' in indicators:
            qvix = indicators['qvix']
            latest = qvix['latest']
            stats = qvix['stats']
            summary_lines.append(
                f"**沪深300期权波动率(QVIX)**: 当前{latest['close']:.2f} "
                f"(开盘:{latest['open']:.2f}, 最高:{latest['high']:.2f}, 最低:{latest['low']:.2f})"
            )
            summary_lines.append(
                f"  - 30日均值:{stats['avg_30d']:.2f}, 30日波动率:{stats['volatility_30d']:.2f}, 30日区间:[{stats['min_30d']:.2f}-{stats['max_30d']:.2f}] - 市场恐慌程度指标"
            )
        
        # 巴菲特指标 - 市场估值水平
        if 'buffett' in indicators:
            buffett = indicators['buffett']
            latest = buffett['latest']
            stats = buffett['stats']
            trend_direction = "上升" if stats['trend_30d'] > 0 else "下降" if stats['trend_30d'] < 0 else "持平"
            
            # 分析分位数水平
            ten_year_percentile = latest['ten_year_percentile'] * 100
            total_percentile = latest['total_percentile'] * 100
            
            # 分位数评级
            if ten_year_percentile <= 20:
                percentile_rating = "极低估值"
            elif ten_year_percentile <= 40:
                percentile_rating = "低估值"
            elif ten_year_percentile <= 60:
                percentile_rating = "合理估值"
            elif ten_year_percentile <= 80:
                percentile_rating = "偏高估值"
            else:
                percentile_rating = "高估值"
            
            summary_lines.append(
                f"**巴菲特指标**: 当前{latest['close_price']:.2f}% "
                f"(总市值:{latest['market_cap']:.0f}万亿, GDP:{latest['gdp']:.0f}万亿)"
            )
            summary_lines.append(
                f"  - 30日均值:{stats['avg_30d']:.2f}%, 30日趋势:{trend_direction}({stats['trend_30d']:+.2f}%), 30日区间:[{stats['min_30d']:.2f}%-{stats['max_30d']:.2f}%]"
            )
            summary_lines.append(
                f"  - **估值水平**: {percentile_rating} (近十年分位数:{ten_year_percentile:.1f}%, 总历史分位数:{total_percentile:.1f}%)"
            )
        
        # 股债利差 - 市场风险偏好
        if 'ebs' in indicators:
            ebs = indicators['ebs']
            latest = ebs['latest']
            stats = ebs['stats']
            trend_direction = "上升" if stats['trend_30d'] > 0 else "下降" if stats['trend_30d'] < 0 else "持平"
            summary_lines.append(
                f"**股债利差**: 当前{latest['stock_bond_spread']:.2f}% "
                f"(沪深300:{latest['hs300_index']:.2f}, 利差均线:{latest['spread_ma']:.2f}%)"
            )
            summary_lines.append(
                f"  - 30日均值:{stats['avg_spread_30d']:.2f}%, 30日趋势:{trend_direction}({stats['trend_30d']:+.2f}%), 30日区间:[{stats['min_spread_30d']:.2f}%-{stats['max_spread_30d']:.2f}%] - 市场风险偏好"
            )
        
        # 大盘拥挤度 - 市场微观结构
        if 'congestion' in indicators:
            congestion = indicators['congestion']
            latest = congestion['latest']
            stats = congestion['stats']
            trend_direction = "上升" if stats['trend_30d'] > 0 else "下降" if stats['trend_30d'] < 0 else "持平"
            summary_lines.append(
                f"**大盘拥挤度**: 当前{latest['congestion']:.2f}% "
                f"(成交额前5%个股占比)"
            )
            summary_lines.append(
                f"  - 30日均值:{stats['avg_30d']:.2f}%, 30日趋势:{trend_direction}({stats['trend_30d']:+.2f}%), 30日区间:[{stats['min_30d']:.2f}%-{stats['max_30d']:.2f}%] - 市场微观结构指标"
            )
        
        return "\n".join(summary_lines)
    
    def generate_kline_charts_base64(self, kline_data: dict, trade_date: str) -> dict:
        """
        生成三大指数K线图并返回base64编码字典
        """
        try:
            if not kline_data:
                logger.warning("K线数据为空，无法生成图表")
                return {}
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
            except:
                pass
            
            charts_base64 = {}
            
            for stock_code, stock_info in kline_data.items():
                stock_name = stock_info['name']
                data_list = stock_info['data']
                
                if not data_list:
                    logger.warning(f"{stock_name}数据不可用，跳过图表生成")
                    continue
                
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                
                df_data = []
                for item in data_list:
                    df_data.append({
                        'date': datetime.strptime(str(item['trade_date']), '%Y%m%d'),
                        'open': item['open_price'],
                        'high': item['high_price'],
                        'low': item['low_price'],
                        'close': item['close_price'],
                        'volume': item['trade_lots']
                    })
                
                df = pd.DataFrame(df_data)
                df = df.sort_values('date')
                
                x_positions = np.arange(len(df))
                
                # 绘制K线
                for j in range(len(df)):
                    open_price = df.iloc[j]['open']
                    high_price = df.iloc[j]['high']
                    low_price = df.iloc[j]['low']
                    close_price = df.iloc[j]['close']
                    
                    if close_price >= open_price:
                        color = '#ff6b6b'  # 上涨红色
                        edge_color = '#ff6b6b'
                    else:
                        color = '#51cf66'  # 下跌绿色
                        edge_color = '#51cf66'
                    
                    ax.plot([j, j], [low_price, high_price], color=edge_color, linewidth=1, alpha=0.8)
                    
                    body_height = abs(close_price - open_price)
                    body_bottom = min(open_price, close_price)
                    
                    if body_height > 0:
                        rect = Rectangle((j - 0.3, body_bottom), 0.6, body_height, 
                                       facecolor=color, edgecolor=edge_color, alpha=0.8, linewidth=0.8)
                        ax.add_patch(rect)
                    else:
                        ax.plot([j, j], [open_price, close_price], color=edge_color, linewidth=2, alpha=0.8)
                
                # 添加移动平均线
                if len(df) >= 5:
                    ma5 = df['close'].rolling(window=5).mean()
                    ax.plot(x_positions, ma5, color='#ffa500', linewidth=1.5, alpha=0.8, label='MA5')
                
                if len(df) >= 10:
                    ma10 = df['close'].rolling(window=10).mean()
                    ax.plot(x_positions, ma10, color='#ff69b4', linewidth=1.5, alpha=0.8, label='MA10')
                
                if len(df) >= 20:
                    ma20 = df['close'].rolling(window=20).mean()
                    ax.plot(x_positions, ma20, color='#4169e1', linewidth=1.5, alpha=0.8, label='MA20')
                
                ax.set_title(f'{stock_name} K线图 - {trade_date}', fontsize=14, fontweight='bold')
                ax.set_ylabel('价格 (点)', fontsize=12)
                ax.set_xlabel('日期', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left', fontsize=10)
                
                if len(df) > 0:
                    step = max(1, len(df) // 8)
                    tick_positions = list(range(0, len(df), step))
                    tick_labels = [df.iloc[i]['date'].strftime('%m-%d') for i in tick_positions if i < len(df)]
                    
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels, rotation=45, fontsize=10)
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                
                charts_base64[stock_code] = {
                    'name': stock_name,
                    'base64': img_base64
                }
                
                logger.info(f"成功生成{stock_name}K线图，大小: {len(img_base64)} 字符")
            
            logger.info(f"成功生成{len(charts_base64)}张K线图")
            return charts_base64
            
        except Exception as e:
            logger.error(f"生成K线图失败: {e}")
            return {}
    
    async def get_llm_summary(self, trade_date: str) -> dict:
        try:
            logger.info(f"获取 {trade_date} 的价格市场LLM分析总结")
            
            # 获取K线数据
            kline_data = self.get_kline_data(trade_date)
            
            # 获取当日数据
            current_day_data = self.get_current_day_data(trade_date)
            
            # 获取板块资金流向摘要
            sector_summary = self.get_sector_summary(trade_date)
            
            # 获取关键市场指标
            market_indicators = self.get_market_indicators(trade_date)
            
            # 生成K线图
            kline_charts_base64 = self.generate_kline_charts_base64(kline_data, trade_date)
            
            has_kline_charts_base64 = bool(kline_charts_base64)
            has_current_day_data = bool(current_day_data)
            has_sector_summary = bool(sector_summary and sector_summary != "无板块资金流向数据")
            has_market_indicators = bool(market_indicators)

            available_sources = has_kline_charts_base64 + has_current_day_data + has_sector_summary + has_market_indicators
            
            if available_sources == 0:
                return {
                    'trade_date': trade_date,
                    'raw_data': "无数据",
                    'llm_summary': "当日无价格市场数据",
                    'data_count': 0,
                    'kline_charts_base64': {}
                }
            
            prompt = f"""
请分析以下A股市场综合数据，并给出专业的宏观市场分析报告（2500字符以内）：

## 数据说明
- 三大指数K线数据：{trade_date}之前90天的历史数据
- 三大指数当日数据：{trade_date}当天的收盘表现
- 板块资金流向：当前实时数据
- 关键市场指标：{trade_date}及最近的关键市场情绪指标

## 一、三大指数当日收盘情况
{self._format_current_day_data(current_day_data, trade_date)}

## 二、板块资金流向
{sector_summary}

## 三、关键市场指标（含90天历史趋势）
{self.format_market_indicators(market_indicators, trade_date)}

## 四、三大指数K线图分析（如果有提供K线图）
请仔细分析提供的三张K线图（上证指数、创业板指、科创50），关注：
- 近期走势趋势（上涨/下跌/震荡）
- 技术指标表现（MA5、MA10、MA20均线）
- 成交量变化特征
- 支撑阻力位情况

## 分析要求

请综合以上信息和K线图，客观描述市场宏观基本面事实：

## 输出要求
- 总结所参考的三大指数收盘情况、K线图技术分析、板块资金流向数据和关键市场指标，并给出当日宏观市场的整体描述
- **对于当日三大指数的收盘价格必须精确到具体点位，不可模糊描述**
- 分析关键市场指标的历史趋势变化：
  - QVIX恐慌程度：关注30日趋势变化和波动率水平
  - 巴菲特指标估值水平：**重点关注分位数评级**，结合绝对比例和历史分位数判断估值水平（极低估值/低估值/合理估值/偏高估值/高估值）
  - 股债利差风险偏好：观察30日利差趋势和风险偏好变化
  - 大盘拥挤度微观结构：跟踪30日拥挤度趋势和结构变化
- 基于K线图分析技术面特征和趋势
- 结合市场指标的历史趋势分析当前市场情绪和风险偏好变化
- 避免主观判断、情绪化描述和未来预测
- 重点突出宏观的客观基本面事实描述和历史趋势分析
- **请把输出的宏观描述严格控制在2500字符以内，不要超过2500字符**

请基于事实数据生成客观的市场描述报告：
"""
            
            if GLOBAL_VISION_LLM and has_kline_charts_base64:
                image_contents = []
                for stock_code, chart_info in kline_charts_base64.items():
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{chart_info['base64']}",
                            "detail": "high"
                        }
                    })
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + image_contents
                }
                
                messages = [
                    {"role": "system", "content": "你是一位资深的金融市场分析师，专长于综合技术分析、资金流向分析和宏观市场判断。请基于多维度数据生成专业的市场分析报告。"},
                    user_message
                ]
                
                response = await GLOBAL_VISION_LLM.a_run(
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2500
                )
            else:
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
                
                messages = [
                    {"role": "system", "content": "你是一位资深的金融市场分析师，专长于综合技术分析、资金流向分析和宏观市场判断。请基于多维度数据生成专业的市场分析报告。"},
                    user_message
                ]
                response = await GLOBAL_LLM.a_run(
                    messages=messages,
                    thinking=False,
                    temperature=0.3,
                    max_tokens=2500
                )
            
            if response and response.content:
                llm_summary = response.content
            else:
                logger.error(f"LLM分析未返回内容")
                llm_summary = "LLM分析失败"
            
            return {
                'trade_date': trade_date,
                'raw_data': prompt,
                'llm_summary': llm_summary,
                'data_count': available_sources,
                'data_sources': {
                    'kline_data': has_kline_charts_base64,
                    'current_day_data': has_current_day_data,
                    'sector_summary': has_sector_summary,
                    'market_indicators': has_market_indicators
                }
            }
                
        except Exception as e:
            traceback.print_exc()
            logger.error(f"获取LLM总结失败: {e}")
            return {
                'trade_date': trade_date,
                'raw_data': "数据获取失败",
                'llm_summary': f"分析失败: {str(e)}",
                'data_count': 0
            }
    
    def _format_current_day_data(self, current_day_data: dict, trade_date: str) -> str:
        if not current_day_data:
            return f"{trade_date} 无三大指数当日数据"
        
        descriptions = []
        
        for stock_code, data in current_day_data.items():
            change_sign = "+" if data['price_change'] >= 0 else ""
            rate_sign = "+" if data['price_change_rate'] >= 0 else ""
            
            desc = f"**{data['name']}** (代码: {stock_code})\n"
            desc += f"- 收盘价: {data['close_price']:.2f}点\n"
            desc += f"- 开盘价: {data['open_price']:.2f}点\n"
            desc += f"- 最高价: {data['high_price']:.2f}点\n"
            desc += f"- 最低价: {data['low_price']:.2f}点\n"
            desc += f"- 涨跌幅: {change_sign}{data['price_change']:.2f}点 ({rate_sign}{data['price_change_rate']*100:.2f}%)\n"
            desc += f"- 成交额: {data['trade_amount']/100000000:.1f}亿元\n"
            desc += f"- 成交量: {data['trade_lots']/10000:.0f}万手"
            
            descriptions.append(desc)
        
        return f"{trade_date}三大指数收盘情况：\n\n" + "\n\n".join(descriptions)

if __name__ == "__main__":
    price_market = PriceMarketAkshare()
    df = asyncio.run(price_market.get_data("2024-08-19 09:00:00"))
    print(df.content.values[0])
