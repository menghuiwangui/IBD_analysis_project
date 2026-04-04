# IBD Literature Analysis

一个完整的炎症性肠病(IBD)文献分析项目，从PubMed自动获取文献并进行AI分析。

## 功能特点
- 🔍 自动从PubMed获取IBD相关文献
- 📊 数据清洗与预处理
- 🤖 AI主题分析（LDA模型）
- 💊 药物、基因、通路实体提取
- 📈 统计分析与可视化

## 快速开始

### 1. 安装依赖

`pip install -r requirements_minimal.txt`

### 2. 配置邮箱
编辑 `config/settings.py`，修改邮箱地址：
EMAIL = "your_email@example.com"  # 修改为你的邮箱

### 3. 运行完整分析

获取500篇文章，分析8个主题

`python main.py --max-results 500 --topics 8`

### 4. 快速测试
`python examples/quick_start.py`

## 命令行参数
- `--query`: PubMed搜索词 (默认: "IBD OR inflammatory bowel disease")
- `--max-results`: 最大获取数量 (默认: 1000)
- `--crawl-only`: 只爬取数据，不分析
- `--analyze-only`: 只分析已存在的数据
- `--topics`: 主题数量 (默认: 10)

## 项目结构


IBD_ANALYSIS/

├── config/              # 配置文件

├── data/               # 数据文件

├── src/                # 源代码

│   ├── data_collection/   # 数据采集

│   └── ai_analysis/       # AI分析

├── examples/           # 示例代码

├── main.py            # 主程序入口

└──  requirements_minimal.txt           # 主程序入口


## 📊 输出结果

### 数据文件
- `data/raw/ibd_articles_*.json` - 原始PubMed数据
- `data/processed/ibd_processed.csv` - 清洗后的数据

### 分析结果
- `data/analysis_results/ibd_with_topics.csv` - 带主题分析的数据
- `data/analysis_results/ibd_with_entities.csv` - 带实体提取的数据
- `data/analysis_results/topics.txt` - 主题关键词
- `data/analysis_results/entity_report.txt` - 实体统计报告
- `data/analysis_results/*.pkl` - 训练好的模型

## ⚙️ 命令行参数

```bash
基本用法

python main.py

自定义搜索词

python main.py --query "Crohn's disease OR ulcerative colitis"

获取更多文章

python main.py --max-results 2000

分析不同数量主题

python main.py --topics 12

跳过爬取，只分析已有数据

python main.py --skip-crawl

```

## 🔍 搜索词建议
- `IBD OR inflammatory bowel disease` (默认)
- `Crohn's disease OR ulcerative colitis`
- `IBD AND treatment`
- `inflammatory bowel disease AND genetics`
- `Crohn's disease AND therapy`

## ⏱️ 预计时间
- 100篇文章: 2-3分钟
- 1000篇文章: 10-15分钟
- 10000篇文章: 1-2小时

## 📈 分析内容
1. **主题建模** - 使用LDA发现研究热点
2. **实体提取** - 提取药物、基因、通路
3. **统计分析** - 计算出现频率
4. **聚类分析** - 自动分组相关文献

## ⚠️ 注意事项
1. PubMed API有频率限制，不要设置太大`max-results`
2. 首次运行会下载约100MB的AI模型
3. 建议从`--max-results 100`开始测试
4. 确保网络能访问PubMed
5. 保持良好的网络状况、
6. IBD相关文献大概600000篇

## 🆘 常见问题
**Q: 报错`ModuleNotFoundError`**  
A: 确保安装了所有依赖：`pip install -r requirements_minimal.txt`

**Q: 获取不到数据**  
A: 检查网络连接，确保能访问`https://eutils.ncbi.nlm.nih.gov`

**Q: 运行很慢**  
A: 减少`--max-results`，或使用`--skip-crawl`分析已有数据

**Q: 内存不足**  
A: 减少`--max-results`和`--topics`参数

使用方法

1.克隆/下载项目：

```bash
git clone <your-repo-url>
cd IBD_ANALYSIS
```