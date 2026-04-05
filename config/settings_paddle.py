# config/settings_paddle.py
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# PubMed API 设置
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_RETRIES = 5
RETRY_DELAY = 2
REQUEST_TIMEOUT = 60
#EMAIL = "your_email@example.com"  # 务必修改
EMAIL = "2964668101@qq.com"
TOOL = "IBD_Paddle_Analysis_v1.0"

# 飞桨平台专用路径
BASE_DATA_DIR = "/home/aistudio/data"  # 飞桨数据目录
if os.path.exists(BASE_DATA_DIR):
    DATA_DIR = Path(BASE_DATA_DIR) / "ibd_analysis"
else:
    DATA_DIR = PROJECT_ROOT / "data"

# 子目录
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "analysis_results"
LOG_DIR = DATA_DIR / "logs"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
MODEL_DIR = DATA_DIR / "models"

# AI 设置
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PADDLE_EMBEDDING_MODEL = "ernie-3.0-medium-zh"  # 中文可选
TOPIC_N_COMPONENTS = 80
MAX_WORDS = 2000
CHUNK_SIZE = 50000  # 分批处理大小

# 飞桨GPU设置
USE_GPU = True
GPU_MEMORY_FRACTION = 0.8