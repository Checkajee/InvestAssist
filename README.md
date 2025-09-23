# 智能投资辅助系统 (InvestAssist)


一个基于大语言模型的智能股票分析系统，集成了多维度数据源、AI分析师辩论和综合决策功能。

## 🚀 项目特色

### 🧠 AI驱动分析
- **双分析师辩论**: 看涨/看跌分析师基于真实数据进行辩论
- **智能决策**: LLM综合辩论结果生成投资决策
- **多模型支持**: 支持DeepSeek、阿里云通义千问等多种LLM

### 📊 多维度数据源
- **价格市场数据**: 三大指数K线、技术指标、市场情绪
- **热钱活跃度**: 涨停跌停、龙虎榜、游资营业部数据
- **新闻资讯**: 新浪财经实时新闻爬取和LLM分析
- **宏观经济**: GDP、CPI、PPI等核心经济指标
- **财务数据**: 个股财务报表深度分析
- **个股分析**: 个股历史行情、市场热度、综合评分、机构参与度

### ⚡ 高效架构
- **异步并发**: 多数据源并行获取，大幅提升响应速度
- **智能缓存**: 基于时间戳的缓存机制，避免重复API调用
- **模块化设计**: 清晰的分层架构，易于扩展和维护

## 📁 项目结构

```
test_model/
├── 📁 analysts/                 # 分析师模块
│   ├── analyst_manager.py      # 分析师管理器（辩论+决策）
│   ├── bull_analyst.py         # 看涨分析师
│   ├── bear_analyst.py         # 看跌分析师
│   └── debate_recorder.py      # 辩论记录器
├── 📁 data_source/             # 数据源模块
│   ├── data_source_base.py     # 数据源基类
│   ├── price_market_akshare.py # 价格市场数据
│   ├── hot_money_akshare.py    # 热钱市场数据
│   ├── sina_news_crawl.py      # 新闻爬虫
│   ├── macro_econo.py          # 宏观经济数据
│   ├── financial_statement_akshare.py # 财务数据
│   └── stock_analysis_akshare.py # 个股分析数据
├── 📁 models/                  # 模型模块
│   ├── llm_model.py           # LLM模型实现
│   └── base_agent_model.py    # 基础智能体模型
├── 📁 utils/                   # 工具模块
│   ├── akshare_utils.py       # AKShare缓存工具
│   ├── date_utils.py          # 日期工具
│   └── llm_utils.py           # LLM工具
├── 📁 config/                  # 配置模块
│   ├── config.py              # 配置加载器
│   └── config.yaml            # 配置文件
├── 📁 data_cache/              # 数据缓存目录
│   ├── price_market_akshare/   # 价格数据缓存
│   ├── hot_money_akshare/      # 热钱数据缓存
│   ├── sina_news_crawl/        # 新闻数据缓存
│   ├── macro_econo/            # 宏观数据缓存
│   ├── financial_statement_akshare/ # 财务数据缓存
│   └── stock_analysis_akshare/ # 个股分析数据缓存
├── 📄 main.py                  # 主程序入口
├── 📄 chat.py                  # 交互式聊天程序
├── 📄 comprehensive_analysis.py # 综合分析模块
└── 📄 requirements.txt         # 依赖包列表
```

## 🎯 核心功能

### 1. 综合市场分析
```python
# 获取多维度市场分析
analyzer = ComprehensiveMarketAnalyzer()
analysis = await analyzer.get_comprehensive_analysis()

# 包含：
# - 宏观经济环境分析
# - 市场技术面分析  
# - 热钱活跃度分析
# - 新闻资讯影响分析
# - 市场情绪和资金流向
```

### 2. AI分析师辩论
```python
# 看涨/看跌分析师辩论
analyst_manager = AnalystManager()
result = await analyst_manager.conduct_full_analysis(symbol="000001")

# 包含：
# - 基于真实数据的多轮辩论
# - 智能决策生成
# - 投资建议和风险提示
```

### 3. 交互式聊天
```bash
python chat.py

# 支持命令：
market     # 综合分析
summary    # 快速摘要  
financial  # 财务查询
analysis   # 投资分析
thinking   # 思考模式
```

### 4. 智能数据源
- **自动缓存**: 基于时间戳的智能缓存
- **并行获取**: 多数据源异步并发
- **LLM分析**: 原始数据自动转换为分析报告
- **容错处理**: 网络异常自动重试

## 🔧 安装配置

### 环境要求
- Python 3.10+
- Windows 10/11 (推荐)

### 快速安装
```bash
# 1. 克隆项目
git clone <repository-url> # 尚未发布
cd test_model

# 2. 运行安装脚本
install.bat

# 3. 配置API密钥
# 编辑 config/config.yaml
# 替换 YOUR_DEEPSEEK_API_KEY 为实际密钥
```

## ⚙️ 配置说明

### 配置文件 (config/config.yaml)
```yaml
# LLM配置
llm:
  base_url: "https://api.deepseek.com"
  api_key: "your-api-key"
  model_name: "deepseek-chat"

# 思考模型配置
llm_thinking:
  base_url: "https://api.deepseek.com"  
  api_key: "your-api-key"
  model_name: "deepseek-reasoner"

# 系统配置
system_language: "中文"
```

### 支持的LLM服务
- **DeepSeek**: 推荐，性价比高
- **阿里云通义千问**: 国内访问稳定
- **OpenAI**: 需要科学上网

## 🚀 使用指南

### 1. 基础使用
```bash
# 运行主程序
python main.py

# 启动交互式聊天
python chat.py
```

### 2. 程序化调用
```python
import asyncio
from comprehensive_analysis import ComprehensiveMarketAnalyzer
from analysts.analyst_manager import AnalystManager

async def main():
    # 综合分析
    analyzer = ComprehensiveMarketAnalyzer()
    analysis = await analyzer.get_comprehensive_analysis()
    
    # 个股分析
    manager = AnalystManager()
    result = await manager.conduct_full_analysis(symbol="000001")
    
    print(analysis['comprehensive_analysis'])
    print(result['decision_result'])

asyncio.run(main())
```

### 3. 交互式查询
```bash
python chat.py

# 自然语言查询示例：
"今天大盘怎么样？"           # 自动调用价格数据
"有什么重要新闻吗？"         # 自动调用新闻数据  
"000001的财务情况如何？"    # 自动调用财务数据
"进行000001的投资分析"      # 启动完整分析流程
```

## 📊 数据源说明

### 价格市场数据 (PriceMarketAkshare)
- **数据来源**: AKShare API
- **包含内容**: 
  - 上证指数、深证成指、创业板指K线数据
  - 技术指标：MA、MACD、RSI等
  - 市场情绪指标：VIX、巴菲特指标等
- **更新频率**: 实时
- **缓存策略**: 小时级缓存

### 热钱市场数据 (HotMoneyAkshare)  
- **数据来源**: AKShare API
- **包含内容**:
  - 涨停跌停股票统计
  - 龙虎榜数据和机构明细
  - 概念板块资金流向
  - 游资营业部活跃度
- **更新频率**: 实时
- **缓存策略**: 小时级缓存

### 新闻资讯 (SinaNewsCrawl)
- **数据来源**: 新浪财经爬虫
- **包含内容**:
  - 实时财经新闻
  - 政策公告解读
  - 市场热点追踪
- **更新频率**: 实时爬取
- **处理方式**: LLM自动分析和摘要

### 宏观经济数据 (MacroEcono)
- **数据来源**: AKShare API
- **包含内容**:
  - GDP、CPI、PPI等核心指标
  - 就业市场数据
  - 进出口贸易数据
- **更新频率**: 按官方发布时间
- **缓存策略**: 日级缓存

### 财务数据 (FinancialStatementAkshare)
- **数据来源**: AKShare API
- **包含内容**:
  - 资产负债表、利润表、现金流量表
  - 财务比率分析
  - 盈利能力评估
- **更新频率**: 按财报发布时间
- **缓存策略**: 日级缓存

### 个股分析数据 (StockAnalysisAkshare)
- **数据来源**: AKShare API
- **包含内容**:
  - 个股历史行情和技术分析
  - 市场热度和关注度指标
  - 综合评分和排名
  - 机构参与度分析
  - 个股相关新闻和公告
- **更新频率**: 实时
- **缓存策略**: 小时级缓存

## 🔄 系统架构

### 数据流架构
```
数据源层 → 缓存层 → 处理层 → LLM分析层 → 决策层
    ↓         ↓        ↓         ↓         ↓
  AKShare   本地缓存  数据清洗   AI分析   投资决策
  新浪财经   时间戳   格式统一   多维度    辩论+决策
```

### 核心组件
- **DataSourceBase**: 数据源基类，提供缓存和统一接口
- **ComprehensiveMarketAnalyzer**: 综合分析器，整合多数据源
- **AnalystManager**: 分析师管理器，协调辩论和决策
- **DebateRecorder**: 辩论记录器，管理辩论流程
- **LLMModel**: LLM模型封装，支持多种服务商

## 🛠️ 开发指南

### 添加新数据源
```python
# 1. 继承DataSourceBase
class NewDataSource(DataSourceBase):
    def __init__(self):
        super().__init__("new_data_source")
    
    async def get_data(self, trigger_time: str) -> pd.DataFrame:
        # 实现数据获取逻辑
        pass

# 2. 实现LLM分析
async def get_llm_summary(self, trade_date: str) -> dict:
    # 实现LLM分析逻辑
    pass
```

### 扩展分析师
```python
# 1. 创建新分析师
class CustomAnalyst:
    async def analyze(self, trigger_time: str, symbol: str):
        # 实现分析逻辑
        pass

# 2. 集成到辩论系统
# 修改DebateRecorder添加新分析师
```

## 📈 性能优化

### 缓存策略
- **时间戳缓存**: 基于小时/日的时间戳避免重复请求
- **智能失效**: 根据数据特性设置不同的缓存时间
- **并行加载**: 多数据源异步并发获取

### 性能指标
- **数据获取**: 4个数据源并行获取，总耗时 < 30秒
- **LLM分析**: 单次分析耗时 < 10秒
- **缓存命中率**: > 80% (相同时间段)
- **内存使用**: < 500MB (包含缓存数据)

---

**⚠️ 免责声明**: 本系统仅用于学习和研究目的，不构成投资建议。投资有风险，决策需谨慎。
