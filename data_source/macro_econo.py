"""
基于 akshare 的宏观经济数据源
整合核心宏观经济指标，生成综合宏观经济分析
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

class MacroEcono(DataSourceBase):
    def __init__(self):
        super().__init__("macro_econo")
        
    async def get_data(self, trigger_time: str, symbol: str = None) -> pd.DataFrame:
        """
        获取宏观经济数据
        
        Args:
            trigger_time: 触发时间
            symbol: 股票代码 (可选，宏观经济数据不依赖特定股票)
        """
        try:
            cache_key = f"{trigger_time}_macro"
            df = self.get_data_cached(cache_key)
            if df is not None:
                return df
            
            trade_date = get_smart_trading_date(trigger_time)     
            logger.info(f"获取 {trade_date} 的宏观经济数据")
            
            llm_summary_dict = await self.get_macro_economic_analysis(trade_date)
            data = [{
                "title": f"{trade_date}:宏观经济数据分析",
                "content": llm_summary_dict["llm_summary"],
                "pub_time": trigger_time,
                "url": None,
                "symbol": "MACRO"
            }]
            
            df = pd.DataFrame(data)
            self.save_data_cached(cache_key, df)
            return df
                
        except Exception as e:
            logger.error(f"获取宏观经济数据失败: {e}")
            return pd.DataFrame()
    
    def get_macro_economic_data(self) -> Dict[str, Any]:
        """
        获取宏观经济数据
        
        Returns:
            Dict: 包含主要宏观经济指标的数据
        """
        try:
            logger.info(f"🔍 开始获取宏观经济数据")
            
            macro_data = {}
            
            # 1. 获取城镇调查失业率
            try:
                logger.debug(f"📊 尝试获取城镇调查失业率...")
                unemployment_data = akshare_cached.run(
                    func_name="macro_china_urban_unemployment",
                    func_kwargs={},
                    verbose=False
                )
                if unemployment_data is not None and not unemployment_data.empty:
                    macro_data['unemployment'] = unemployment_data
                    logger.info(f"✅ 成功获取城镇调查失业率: {len(unemployment_data)}条记录")
                    logger.debug(f"失业率列名: {list(unemployment_data.columns)}")
                else:
                    logger.warning(f"⚠️ 城镇调查失业率数据为空")
            except Exception as e:
                logger.warning(f"❌ 获取城镇调查失业率失败: {e}")
            
            # 2. 获取社会融资规模增量统计
            try:
                logger.debug(f"💰 尝试获取社会融资规模增量统计...")
                social_financing_data = akshare_cached.run(
                    func_name="macro_china_shrzgm",
                    func_kwargs={},
                    verbose=False
                )
                if social_financing_data is not None and not social_financing_data.empty:
                    macro_data['social_financing'] = social_financing_data
                    logger.info(f"✅ 成功获取社会融资规模增量: {len(social_financing_data)}条记录")
                    logger.debug(f"社会融资列名: {list(social_financing_data.columns)}")
                else:
                    logger.warning(f"⚠️ 社会融资规模增量数据为空")
            except Exception as e:
                logger.warning(f"❌ 获取社会融资规模增量失败: {e}")
            
            # 3. 获取中国 GDP 年率
            try:
                logger.debug(f"📈 尝试获取中国GDP年率...")
                gdp_data = akshare_cached.run(
                    func_name="macro_china_gdp_yearly",
                    func_kwargs={},
                    verbose=False
                )
                if gdp_data is not None and not gdp_data.empty:
                    macro_data['gdp'] = gdp_data
                    logger.info(f"✅ 成功获取GDP年率: {len(gdp_data)}条记录")
                    logger.debug(f"GDP列名: {list(gdp_data.columns)}")
                else:
                    logger.warning(f"⚠️ GDP年率数据为空")
            except Exception as e:
                logger.warning(f"❌ 获取GDP年率失败: {e}")
            
            # 4. 获取中国 CPI 月率报告
            try:
                logger.debug(f"📊 尝试获取中国CPI月率...")
                cpi_data = akshare_cached.run(
                    func_name="macro_china_cpi_monthly",
                    func_kwargs={},
                    verbose=False
                )
                if cpi_data is not None and not cpi_data.empty:
                    macro_data['cpi'] = cpi_data
                    logger.info(f"✅ 成功获取CPI月率: {len(cpi_data)}条记录")
                    logger.debug(f"CPI列名: {list(cpi_data.columns)}")
                else:
                    logger.warning(f"⚠️ CPI月率数据为空")
            except Exception as e:
                logger.warning(f"❌ 获取CPI月率失败: {e}")
            
            # 5. 获取中国 PPI 年率报告
            try:
                logger.debug(f"🏭 尝试获取中国PPI年率...")
                ppi_data = akshare_cached.run(
                    func_name="macro_china_ppi_yearly",
                    func_kwargs={},
                    verbose=False
                )
                if ppi_data is not None and not ppi_data.empty:
                    macro_data['ppi'] = ppi_data
                    logger.info(f"✅ 成功获取PPI年率: {len(ppi_data)}条记录")
                    logger.debug(f"PPI列名: {list(ppi_data.columns)}")
                else:
                    logger.warning(f"⚠️ PPI年率数据为空")
            except Exception as e:
                logger.warning(f"❌ 获取PPI年率失败: {e}")
            
            # 6. 获取工业增加值增长
            try:
                logger.debug(f"🏭 尝试获取工业增加值增长...")
                industrial_data = akshare_cached.run(
                    func_name="macro_china_gyzjz",
                    func_kwargs={},
                    verbose=False
                )
                if industrial_data is not None and not industrial_data.empty:
                    macro_data['industrial_value'] = industrial_data
                    logger.info(f"✅ 成功获取工业增加值增长: {len(industrial_data)}条记录")
                    logger.debug(f"工业增加值列名: {list(industrial_data.columns)}")
                else:
                    logger.warning(f"⚠️ 工业增加值增长数据为空")
            except Exception as e:
                logger.warning(f"❌ 获取工业增加值增长失败: {e}")
            
            # 7. 获取财新制造业PMI终值
            try:
                logger.debug(f"📊 尝试获取财新制造业PMI...")
                pmi_data = akshare_cached.run(
                    func_name="macro_china_cx_pmi_yearly",
                    func_kwargs={},
                    verbose=False
                )
                if pmi_data is not None and not pmi_data.empty:
                    macro_data['pmi'] = pmi_data
                    logger.info(f"✅ 成功获取财新制造业PMI: {len(pmi_data)}条记录")
                    logger.debug(f"PMI列名: {list(pmi_data.columns)}")
                else:
                    logger.warning(f"⚠️ 财新制造业PMI数据为空")
            except Exception as e:
                logger.warning(f"❌ 获取财新制造业PMI失败: {e}")
            
            # 8. 获取企业景气及企业家信心指数
            try:
                logger.debug(f"📈 尝试获取企业景气及企业家信心指数...")
                enterprise_data = akshare_cached.run(
                    func_name="macro_china_enterprise_boom_index",
                    func_kwargs={},
                    verbose=False
                )
                if enterprise_data is not None and not enterprise_data.empty:
                    macro_data['enterprise_boom'] = enterprise_data
                    logger.info(f"✅ 成功获取企业景气指数: {len(enterprise_data)}条记录")
                    logger.debug(f"企业景气指数列名: {list(enterprise_data.columns)}")
                else:
                    logger.warning(f"⚠️ 企业景气指数数据为空")
            except Exception as e:
                logger.warning(f"❌ 获取企业景气指数失败: {e}")
            
            # 9. 获取以美元计算进口年率
            try:
                logger.debug(f"📦 尝试获取进口年率...")
                imports_data = akshare_cached.run(
                    func_name="macro_china_imports_yoy",
                    func_kwargs={},
                    verbose=False
                )
                if imports_data is not None and not imports_data.empty:
                    macro_data['imports'] = imports_data
                    logger.info(f"✅ 成功获取进口年率: {len(imports_data)}条记录")
                    logger.debug(f"进口年率列名: {list(imports_data.columns)}")
                else:
                    logger.warning(f"⚠️ 进口年率数据为空")
            except Exception as e:
                logger.warning(f"❌ 获取进口年率失败: {e}")
            
            # 10. 获取以美元计算出口年率
            try:
                logger.debug(f"🚢 尝试获取出口年率...")
                exports_data = akshare_cached.run(
                    func_name="macro_china_exports_yoy",
                    func_kwargs={},
                    verbose=False
                )
                if exports_data is not None and not exports_data.empty:
                    macro_data['exports'] = exports_data
                    logger.info(f"✅ 成功获取出口年率: {len(exports_data)}条记录")
                    logger.debug(f"出口年率列名: {list(exports_data.columns)}")
                else:
                    logger.warning(f"⚠️ 出口年率数据为空")
            except Exception as e:
                logger.warning(f"❌ 获取出口年率失败: {e}")
            
            # 记录最终结果
            if macro_data:
                logger.info(f"✅ 宏观经济数据获取完成，包含{len(macro_data)}个数据集")
                for key, value in macro_data.items():
                    if hasattr(value, '__len__'):
                        logger.info(f"  - {key}: {len(value)}条记录")
            else:
                logger.warning(f"⚠️ 未能获取任何宏观经济数据")
            
            return macro_data
            
        except Exception as e:
            logger.error(f"❌ 获取宏观经济数据失败: {e}")
            return {}
    
    def format_macro_economic_data(self, macro_data: Dict[str, Any]) -> str:
        """
        格式化宏观经济数据为文本
        """
        if not macro_data:
            return "无宏观经济数据"
        
        formatted_text = f"📊 宏观经济数据分析:\n\n"
        
        # 1. 城镇调查失业率
        if 'unemployment' in macro_data:
            unemployment_data = macro_data['unemployment']
            if not unemployment_data.empty:
                formatted_text += "## 城镇调查失业率:\n\n"
                
                # 显示最新数据
                recent_data = unemployment_data.tail(5)
                for _, row in recent_data.iterrows():
                    try:
                        date = row.get('date', 'N/A')
                        item = row.get('item', 'N/A')
                        value = row.get('value', 'N/A')
                        
                        formatted_text += f"日期: {date}\n"
                        formatted_text += f"  项目: {item}\n"
                        formatted_text += f"  数值: {value}%\n\n"
                    except Exception as e:
                        logger.warning(f"格式化失业率数据行失败: {e}")
                        continue
        
        # 2. 社会融资规模增量统计
        if 'social_financing' in macro_data:
            social_financing_data = macro_data['social_financing']
            if not social_financing_data.empty:
                formatted_text += "## 社会融资规模增量统计:\n\n"
                
                # 显示最新数据
                recent_data = social_financing_data.tail(3)
                for _, row in recent_data.iterrows():
                    try:
                        month = row.get('月份', 'N/A')
                        total_increment = row.get('社会融资规模增量', 'N/A')
                        rmb_loan = row.get('其中-人民币贷款', 'N/A')
                        
                        formatted_text += f"月份: {month}\n"
                        formatted_text += f"  社会融资规模增量: {total_increment}亿元\n"
                        formatted_text += f"  其中-人民币贷款: {rmb_loan}亿元\n\n"
                    except Exception as e:
                        logger.warning(f"格式化社会融资数据行失败: {e}")
                        continue
        
        # 3. 中国 GDP 年率
        if 'gdp' in macro_data:
            gdp_data = macro_data['gdp']
            if not gdp_data.empty:
                formatted_text += "## 中国GDP年率:\n\n"
                
                # 显示最新数据
                recent_data = gdp_data.tail(3)
                for _, row in recent_data.iterrows():
                    try:
                        commodity = row.get('商品', 'N/A')
                        date = row.get('日期', 'N/A')
                        current_value = row.get('今值', 'N/A')
                        forecast_value = row.get('预测值', 'N/A')
                        previous_value = row.get('前值', 'N/A')
                        
                        formatted_text += f"商品: {commodity}\n"
                        formatted_text += f"  日期: {date}\n"
                        formatted_text += f"  今值: {current_value}%\n"
                        formatted_text += f"  预测值: {forecast_value}%\n"
                        formatted_text += f"  前值: {previous_value}%\n\n"
                    except Exception as e:
                        logger.warning(f"格式化GDP数据行失败: {e}")
                        continue
        
        # 4. 中国 CPI 月率报告
        if 'cpi' in macro_data:
            cpi_data = macro_data['cpi']
            if not cpi_data.empty:
                formatted_text += "## 中国CPI月率报告:\n\n"
                
                # 显示最新数据
                recent_data = cpi_data.tail(3)
                for _, row in recent_data.iterrows():
                    try:
                        commodity = row.get('商品', 'N/A')
                        date = row.get('日期', 'N/A')
                        current_value = row.get('今值', 'N/A')
                        forecast_value = row.get('预测值', 'N/A')
                        previous_value = row.get('前值', 'N/A')
                        
                        formatted_text += f"商品: {commodity}\n"
                        formatted_text += f"  日期: {date}\n"
                        formatted_text += f"  今值: {current_value}%\n"
                        formatted_text += f"  预测值: {forecast_value}%\n"
                        formatted_text += f"  前值: {previous_value}%\n\n"
                    except Exception as e:
                        logger.warning(f"格式化CPI数据行失败: {e}")
                        continue
        
        # 5. 中国 PPI 年率报告
        if 'ppi' in macro_data:
            ppi_data = macro_data['ppi']
            if not ppi_data.empty:
                formatted_text += "## 中国PPI年率报告:\n\n"
                
                # 显示最新数据
                recent_data = ppi_data.tail(3)
                for _, row in recent_data.iterrows():
                    try:
                        commodity = row.get('商品', 'N/A')
                        date = row.get('日期', 'N/A')
                        current_value = row.get('今值', 'N/A')
                        forecast_value = row.get('预测值', 'N/A')
                        previous_value = row.get('前值', 'N/A')
                        
                        formatted_text += f"商品: {commodity}\n"
                        formatted_text += f"  日期: {date}\n"
                        formatted_text += f"  今值: {current_value}%\n"
                        formatted_text += f"  预测值: {forecast_value}%\n"
                        formatted_text += f"  前值: {previous_value}%\n\n"
                    except Exception as e:
                        logger.warning(f"格式化PPI数据行失败: {e}")
                        continue
        
        # 6. 工业增加值增长
        if 'industrial_value' in macro_data:
            industrial_data = macro_data['industrial_value']
            if not industrial_data.empty:
                formatted_text += "## 工业增加值增长:\n\n"
                
                # 显示最新数据
                recent_data = industrial_data.tail(3)
                for _, row in recent_data.iterrows():
                    try:
                        month = row.get('月份', 'N/A')
                        yoy_growth = row.get('同比增长', 'N/A')
                        cumulative_growth = row.get('累计增长', 'N/A')
                        release_time = row.get('发布时间', 'N/A')
                        
                        formatted_text += f"月份: {month}\n"
                        formatted_text += f"  同比增长: {yoy_growth}%\n"
                        formatted_text += f"  累计增长: {cumulative_growth}%\n"
                        formatted_text += f"  发布时间: {release_time}\n\n"
                    except Exception as e:
                        logger.warning(f"格式化工业增加值数据行失败: {e}")
                        continue
        
        # 7. 财新制造业PMI终值
        if 'pmi' in macro_data:
            pmi_data = macro_data['pmi']
            if not pmi_data.empty:
                formatted_text += "## 财新制造业PMI终值:\n\n"
                
                # 显示最新数据
                recent_data = pmi_data.tail(3)
                for _, row in recent_data.iterrows():
                    try:
                        commodity = row.get('商品', 'N/A')
                        date = row.get('日期', 'N/A')
                        current_value = row.get('今值', 'N/A')
                        forecast_value = row.get('预测值', 'N/A')
                        previous_value = row.get('前值', 'N/A')
                        
                        formatted_text += f"商品: {commodity}\n"
                        formatted_text += f"  日期: {date}\n"
                        formatted_text += f"  今值: {current_value}\n"
                        formatted_text += f"  预测值: {forecast_value}\n"
                        formatted_text += f"  前值: {previous_value}\n\n"
                    except Exception as e:
                        logger.warning(f"格式化PMI数据行失败: {e}")
                        continue
        
        # 8. 企业景气及企业家信心指数
        if 'enterprise_boom' in macro_data:
            enterprise_data = macro_data['enterprise_boom']
            if not enterprise_data.empty:
                formatted_text += "## 企业景气及企业家信心指数:\n\n"
                
                # 显示最新数据
                recent_data = enterprise_data.tail(3)
                for _, row in recent_data.iterrows():
                    try:
                        quarter = row.get('季度', 'N/A')
                        boom_index = row.get('企业景气指数-指数', 'N/A')
                        boom_yoy = row.get('企业景气指数-同比', 'N/A')
                        boom_mom = row.get('企业景气指数-环比', 'N/A')
                        confidence_index = row.get('企业家信心指数-指数', 'N/A')
                        confidence_yoy = row.get('企业家信心指数-同比', 'N/A')
                        confidence_mom = row.get('企业家信心指数-环比', 'N/A')
                        
                        formatted_text += f"季度: {quarter}\n"
                        formatted_text += f"  企业景气指数: {boom_index}\n"
                        formatted_text += f"    同比: {boom_yoy}%\n"
                        formatted_text += f"    环比: {boom_mom}%\n"
                        formatted_text += f"  企业家信心指数: {confidence_index}\n"
                        formatted_text += f"    同比: {confidence_yoy}%\n"
                        formatted_text += f"    环比: {confidence_mom}%\n\n"
                    except Exception as e:
                        logger.warning(f"格式化企业景气数据行失败: {e}")
                        continue
        
        # 9. 以美元计算进口年率
        if 'imports' in macro_data:
            imports_data = macro_data['imports']
            if not imports_data.empty:
                formatted_text += "## 以美元计算进口年率:\n\n"
                
                # 显示最新数据
                recent_data = imports_data.tail(3)
                for _, row in recent_data.iterrows():
                    try:
                        commodity = row.get('商品', 'N/A')
                        date = row.get('日期', 'N/A')
                        current_value = row.get('今值', 'N/A')
                        forecast_value = row.get('预测值', 'N/A')
                        previous_value = row.get('前值', 'N/A')
                        
                        formatted_text += f"商品: {commodity}\n"
                        formatted_text += f"  日期: {date}\n"
                        formatted_text += f"  今值: {current_value}%\n"
                        formatted_text += f"  预测值: {forecast_value}%\n"
                        formatted_text += f"  前值: {previous_value}%\n\n"
                    except Exception as e:
                        logger.warning(f"格式化进口数据行失败: {e}")
                        continue
        
        # 10. 以美元计算出口年率
        if 'exports' in macro_data:
            exports_data = macro_data['exports']
            if not exports_data.empty:
                formatted_text += "## 以美元计算出口年率:\n\n"
                
                # 显示最新数据
                recent_data = exports_data.tail(3)
                for _, row in recent_data.iterrows():
                    try:
                        commodity = row.get('商品', 'N/A')
                        date = row.get('日期', 'N/A')
                        current_value = row.get('今值', 'N/A')
                        forecast_value = row.get('预测值', 'N/A')
                        previous_value = row.get('前值', 'N/A')
                        
                        formatted_text += f"商品: {commodity}\n"
                        formatted_text += f"  日期: {date}\n"
                        formatted_text += f"  今值: {current_value}%\n"
                        formatted_text += f"  预测值: {forecast_value}%\n"
                        formatted_text += f"  前值: {previous_value}%\n\n"
                    except Exception as e:
                        logger.warning(f"格式化出口数据行失败: {e}")
                        continue
        
        return formatted_text
    
    async def get_macro_economic_analysis(self, trade_date: str) -> dict:
        """
        获取宏观经济分析
        """
        try:
            logger.info(f"获取 {trade_date} 的宏观经济分析")
            
            # 获取宏观经济数据
            macro_data = self.get_macro_economic_data()
            
            if not macro_data:
                return {
                    'trade_date': trade_date,
                    'symbol': 'MACRO',
                    'raw_data': "无宏观经济数据",
                    'llm_summary': f"{trade_date} 无可用宏观经济数据",
                    'data_count': 0
                }
            
            # 格式化宏观经济数据
            formatted_data = self.format_macro_economic_data(macro_data)
            
            # 构建LLM分析提示词
            prompt = f"""
请分析以下宏观经济数据，并给出专业的宏观经济分析报告（2000字符以内）：

## 宏观经济数据汇总
{formatted_data}

## 分析要求
请基于提供的宏观经济数据，进行以下维度的分析：

1. **经济增长分析**：
   - 分析GDP增长趋势和经济增长动力
   - 评估工业增加值增长情况
   - 分析经济增长的质量和可持续性

2. **通胀压力分析**：
   - 分析CPI和PPI的变化趋势
   - 评估通胀水平和通胀预期
   - 分析通胀对经济的影响

3. **就业市场分析**：
   - 分析城镇调查失业率变化
   - 评估就业市场状况
   - 分析就业与经济增长的关系

4. **货币政策分析**：
   - 分析社会融资规模变化
   - 评估货币供应量变化
   - 分析货币政策对经济的影响

5. **制造业分析**：
   - 分析财新制造业PMI变化
   - 评估制造业景气度
   - 分析制造业对经济的影响

6. **企业信心分析**：
   - 分析企业景气指数变化
   - 评估企业家信心指数
   - 分析企业投资意愿

7. **对外贸易分析**：
   - 分析进出口增长率变化
   - 评估对外贸易状况
   - 分析外需对经济的影响

8. **综合评估**：
   - 综合评估宏观经济整体状况
   - 分析各指标间的关联性
   - 判断经济周期位置和趋势

## 输出要求
- 基于实际宏观经济数据进行分析
- 保持客观专业，避免主观判断
- 突出关键指标和变化趋势
- 控制在2000字符以内
"""
            
            messages = [
                {
                    "role": "system", 
                    "content": "你是一位资深的宏观经济分析师，专长于宏观经济指标分析和经济趋势判断。请基于实际宏观经济数据生成专业的分析报告。"
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
                logger.error(f"LLM宏观经济分析未返回内容")
                llm_summary = f"{trade_date} 宏观经济分析失败"
            
            return {
                'trade_date': trade_date,
                'symbol': 'MACRO',
                'raw_data': formatted_data,
                'llm_summary': llm_summary,
                'data_count': len(macro_data)
            }
                
        except Exception as e:
            traceback.print_exc()
            logger.error(f"获取宏观经济分析失败: {e}")
            return {
                'trade_date': trade_date,
                'symbol': 'MACRO',
                'raw_data': "数据获取失败",
                'llm_summary': f"{trade_date} 宏观经济分析失败: {str(e)}",
                'data_count': 0
            }


if __name__ == "__main__":
    # 测试宏观经济数据
    macro_analyzer = MacroEcono()
    df = asyncio.run(macro_analyzer.get_data("2024-08-19 09:00:00"))
    print(df.content.values[0] if not df.empty else "无数据")
