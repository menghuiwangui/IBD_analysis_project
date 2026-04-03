# IBD 研究分析项目

## 项目简介

本项目使用 pebmeb API 获取 IBD（炎症性肠病）相关文献，并使用飞浆（PaddlePaddle）平台进行 AI 分析，以了解 IBD 前沿研究方向，包括治疗方法、相关药物、基因点位和治疗通路等。

## 项目结构

```
IBD_analysis/
├── src/
│   ├── __init__.py
│   ├── api.py         # API调用模块
│   ├── data_processor.py  # 数据处理模块
│   ├── ai_analyzer.py     # AI分析模块
│   └── report_generator.py  # 报告生成模块
├── main.py            # 主脚本
├── requirements.txt   # 依赖文件
├── .env               # 环境变量配置
└── README.md          # 项目说明
```

## 环境配置

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置邮箱地址

在 `.env` 文件中填写 pebmeb 邮箱地址：

```
PEBMEB_EMAIL=your_email_here
```

## 使用方法

运行主脚本开始分析：

```bash
python main.py
```

## 功能说明

1. **API 调用**：从 pebmeb API 获取所有 IBD 相关文献
2. **数据处理**：将文献数据保存为 JSON 和 CSV 文件
3. **AI 分析**：使用飞浆平台分析文献中的治疗方法、药物、基因和通路
4. **报告生成**：生成分析报告和可视化图表

## 分析结果

分析结果将保存在以下位置：

- `data/` 目录：原始数据和处理后的数据
- `reports/` 目录：分析报告和可视化图表

## 注意事项

- 请确保网络连接正常，以便能够访问 pebmeb API
- API 调用可能会受到速率限制，请耐心等待
- 对于大量文献数据，分析过程可能需要较长时间
