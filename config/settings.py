# config/settings.py
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# PubMed API 设置
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_RETRIES = 3
RETRY_DELAY = 2
REQUEST_TIMEOUT = 30
#EMAIL = "your_email@example.com"  # 改成你的邮箱
EMAIL = "2964668101@qq.com"
TOOL = "IBD_Analysis"

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "analysis_results"
LOG_DIR = DATA_DIR / "logs"

# 创建目录
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# AI 设置
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOPIC_N_COMPONENTS = 10
MAX_WORDS = 1000