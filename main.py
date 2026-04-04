# main.py
import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.settings import EMAIL
from src.data_collection import PubMedClient, process_data
from src.ai_analysis import TopicAnalyzer, DrugGeneExtractor

def main():
    parser = argparse.ArgumentParser(description="IBD Literature Analysis Pipeline")
    parser.add_argument("--query", type=str, default="IBD OR inflammatory bowel disease",
                       help="PubMed search query")
    parser.add_argument("--max-results", type=int, default=500,
                       help="Maximum number of articles to fetch")
    parser.add_argument("--topics", type=int, default=8,
                       help="Number of topics for LDA")
    parser.add_argument("--skip-crawl", action="store_true",
                       help="Skip crawling, use existing data")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IBD LITERATURE ANALYSIS PIPELINE")
    print("=" * 60)
    
    if not args.skip_crawl:
        # Step 1: 获取数据
        print("\n[1/4] COLLECTING DATA FROM PUBMED")
        print("-" * 40)
        
        client = PubMedClient(email=EMAIL)
        raw_file, articles = client.crawl(
            query=args.query,
            max_results=args.max_results
        )
        
        if not articles:
            print("❌ No articles downloaded. Exiting.")
            return
        
        # Step 2: 处理数据
        print(f"\n[2/4] PROCESSING {len(articles)} ARTICLES")
        print("-" * 40)
        
        if raw_file:
            df_processed = process_data(
                Path(raw_file).name,
                "ibd_processed.csv"
            )
        else:
            print("❌ No data file. Exiting.")
            return
    else:
        # 加载已有数据
        print("\n[1/4] SKIPPING CRAWL")
        print("[2/4] LOADING EXISTING DATA")
        print("-" * 40)
        
        from config.settings import PROCESSED_DATA_DIR
        processed_file = PROCESSED_DATA_DIR / "ibd_processed.csv"
        
        if not processed_file.exists():
            print(f"❌ Processed data not found: {processed_file}")
            print("   Run without --skip-crawl first")
            return
        
        import pandas as pd
        df_processed = pd.read_csv(processed_file)
        print(f"✅ Loaded {len(df_processed)} articles from {processed_file}")
    
    if df_processed.empty:
        print("❌ No data to analyze. Exiting.")
        return
    
    # Step 3: 主题分析
    print(f"\n[3/4] TOPIC ANALYSIS ({args.topics} TOPICS)")
    print("-" * 40)
    
    try:
        topic_analyzer = TopicAnalyzer(n_topics=args.topics)
        df_with_topics, topics = topic_analyzer.analyze(df_processed, save=True)
        
        print("\n📋 TOPIC KEYWORDS:")
        for i, topic_words in enumerate(topics):
            print(f"  Topic {i}: {', '.join(topic_words[:5])}")
        
    except Exception as e:
        print(f"❌ Topic analysis failed: {e}")
        df_with_topics = df_processed
    
    # Step 4: 实体提取
    print(f"\n[4/4] EXTRACTING DRUGS, GENES & PATHWAYS")
    print("-" * 40)
    
    try:
        extractor = DrugGeneExtractor()
        df_with_entities = extractor.analyze(df_with_topics)
        stats = extractor.get_statistics(df_with_entities)
        extractor.save_results(df_with_entities, stats)
        
        # 打印摘要
        print("\n📊 ANALYSIS SUMMARY:")
        print(f"  Articles analyzed: {len(df_with_entities)}")
        print(f"  Unique drugs mentioned: {len(stats['drugs'])}")
        print(f"  Unique genes mentioned: {len(stats['genes'])}")
        print(f"  Unique pathways mentioned: {len(stats['pathways'])}")
        
        print("\n🏆 TOP FINDINGS:")
        print("  Most mentioned drugs:")
        for drug, count in stats['drugs'][:3]:
            print(f"    • {drug}: {count} articles")
        
        print("  Most mentioned genes:")
        for gene, count in stats['genes'][:3]:
            print(f"    • {gene}: {count} articles")
        
        print("  Most mentioned pathways:")
        for pathway, count in stats['pathways'][:3]:
            print(f"    • {pathway}: {count} articles")
        
    except Exception as e:
        print(f"❌ Entity extraction failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print("\n📁 Results saved in:")
    print("  data/processed/ - Cleaned data")
    print("  data/analysis_results/ - Analysis results")

if __name__ == "__main__":
    main()