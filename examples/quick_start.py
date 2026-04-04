# examples/quick_start.py
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from main import main

if __name__ == "__main__":
    print("🚀 Quick Start: IBD Literature Analysis")
    print("Fetching 100 articles for testing...\n")
    
    # 只获取100篇文章，分析5个主题
    main(
        query="IBD",
        max_results=100,
        topics=5,
        skip_crawl=False
    )