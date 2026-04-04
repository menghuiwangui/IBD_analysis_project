# paddle_main.py
"""
飞桨平台专用主程序 - 增强版
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 飞桨平台特殊设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.settings_paddle import *
from src.data_collection.pubmed_client_enhanced import PubMedEnhancedClient
from src.data_collection.data_processor_enhanced import EnhancedDataProcessor
from src.ai_analysis.topic_model_paddle import PaddleTopicAnalyzer
from src.ai_analysis.entity_extractor_enhanced import EnhancedEntityExtractor

# ✅ 导入优化模块
from src.utils.memory_optimizer import MemoryOptimizer
from src.utils.progress_tracker import ProgressTracker
from src.utils.resource_monitor import ResourceMonitor

class PaddleIBDAnalyzerEnhanced:
    """增强版飞桨平台分析器"""
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.setup_directories()
        
        # ✅ 初始化优化工具
        self.memory_optimizer = MemoryOptimizer()
        self.progress_tracker = ProgressTracker(CHECKPOINT_DIR, "ibd_analysis")
        self.resource_monitor = ResourceMonitor(LOG_DIR, interval=300)  # 5分钟监控一次
        
    def setup_directories(self):
        """创建目录结构"""
        dirs = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                RESULTS_DIR, LOG_DIR, CHECKPOINT_DIR]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ 目录已创建: {dir_path}")
    
    def start_monitoring(self):
        """启动资源监控"""
        self.resource_monitor.start()
    
    def stop_monitoring(self):
        """停止资源监控"""
        self.resource_monitor.stop()
        
        # 生成监控报告
        report = self.resource_monitor.generate_report()
        report_file = LOG_DIR / "resource_monitor_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 监控报告已生成: {report_file}")
    
    def crawl_in_batches_optimized(self, query, total_count=600000, batch_size=10000):
        """优化版分批获取文献"""
        print(f"🚀 开始分批获取IBD文献，目标: {total_count:,}篇")
        
        # 启动监控
        self.start_monitoring()
        
        # 记录开始状态
        self.progress_tracker.save_state({
            "start_time": datetime.now().isoformat(),
            "task": "crawl",
            "total": total_count,
            "current": 0,
            "status": "started"
        })
        
        client = PubMedEnhancedClient(email=EMAIL, tool=TOOL)
        
        all_articles = []
        articles_processed = 0
        
        try:
            # 分多个搜索词并行获取
            search_terms = self._get_search_strategies()
            
            for i, (term, max_results) in enumerate(search_terms):
                if articles_processed >= total_count:
                    break
                
                print(f"\n🔍 策略 {i+1}/{len(search_terms)}: {term}")
                
                # 更新进度
                self.progress_tracker.log_step(f"search_{i}", {
                    "term": term,
                    "max_results": max_results
                })
                
                # 获取文献ID
                remaining = total_count - articles_processed
                actual_max = min(max_results, remaining)
                
                id_list = client.search_ids_comprehensive(term, max_results=actual_max)
                
                if not id_list:
                    print(f"⚠️  无结果: {term}")
                    continue
                
                # 分批处理
                for batch_idx, start_idx in enumerate(range(0, len(id_list), batch_size)):
                    batch_ids = id_list[start_idx:start_idx + batch_size]
                    
                    # 获取批次
                    articles = client._fetch_batch(batch_ids)
                    all_articles.extend(articles)
                    articles_processed += len(articles)
                    
                    # ✅ 优化：定期清理内存
                    if len(all_articles) % 50000 == 0:
                        self._process_batch_checkpoint(all_articles, articles_processed)
                    
                    # 更新进度
                    self.progress_tracker.save_state({
                        "total": total_count,
                        "current": articles_processed,
                        "status": "crawling",
                        "current_term": term,
                        "current_batch": batch_idx
                    })
                    
                    # 进度显示
                    print(f"  已获取: {articles_processed:,}/{total_count:,} "
                          f"({articles_processed/total_count*100:.1f}%)")
                    
                    if articles_processed >= total_count:
                        break
                    
                    # 礼貌延迟
                    time.sleep(0.2)
        
        except KeyboardInterrupt:
            print("\n⏸️ 爬取被中断，保存当前进度...")
        except Exception as e:
            print(f"❌ 爬取出错: {e}")
        finally:
            # 保存最终结果
            if all_articles:
                self._save_final_results(all_articles, client)
            
            # 停止监控
            self.stop_monitoring()
            
            # 保存最终状态
            self.progress_tracker.save_state({
                "total": total_count,
                "current": articles_processed,
                "status": "completed" if articles_processed >= total_count else "interrupted",
                "end_time": datetime.now().isoformat()
            })
        
        return all_articles
    
    def _process_batch_checkpoint(self, articles, count):
        """处理批次检查点"""
        # ✅ 优化：减少内存占用
        if len(articles) > 100000:
            # 保存检查点
            self.progress_tracker.save_checkpoint(
                articles[-50000:],  # 只保存最近5万篇
                f"batch_{count}",
                "articles"
            )
            
            # 清理内存
            self.memory_optimizer.clear_memory()
            
            # 打印内存使用
            self.memory_optimizer.print_memory_usage(f"批次 {count}")
    
    def _save_final_results(self, articles, client):
        """保存最终结果"""
        # ✅ 优化：分批保存大文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 如果数据太大，分多个文件保存
        if len(articles) > 200000:
            for i in range(0, len(articles), 200000):
                chunk = articles[i:i + 200000]
                filename = f"ibd_articles_{i//200000 + 1}_{len(chunk)}_{timestamp}.json"
                client.save_data(chunk, filename)
        else:
            filename = f"ibd_articles_{len(articles)}_{timestamp}.json"
            client.save_data(articles, filename)
    
    def _get_search_strategies(self):
        """获取搜索策略"""
        return [
            ("IBD[MeSH]", 200000),
            ("inflammatory bowel disease[MeSH]", 200000),
            ("Crohn disease[MeSH]", 100000),
            ("ulcerative colitis[MeSH]", 100000),
            ("IBD OR inflammatory bowel disease", 100000)
        ]
    
    def process_large_data_optimized(self, raw_files, chunk_size=50000):
        """优化版大规模数据处理"""
        print(f"\n🔧 开始优化处理大规模数据")
        
        processor = EnhancedDataProcessor()
        all_chunks = []
        
        # ✅ 使用内存优化处理
        for raw_file in raw_files:
            print(f"📁 处理文件: {raw_file}")
            
            # 分块读取和处理
            with open(raw_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 分批处理
            for i in tqdm(range(0, len(data), chunk_size), desc="处理批次"):
                chunk_data = data[i:i + chunk_size]
                
                # 处理并优化内存
                chunk_df = processor.clean_data(chunk_data)
                chunk_df_optimized = self.memory_optimizer.optimize_dataframe(chunk_df)
                
                if not chunk_df_optimized.empty:
                    all_chunks.append(chunk_df_optimized)
                
                # 清理内存
                del chunk_data, chunk_df, chunk_df_optimized
                self.memory_optimizer.clear_memory()
        
        # 合并结果
        if all_chunks:
            # ✅ 使用增量合并避免内存溢出
            final_df = pd.concat(all_chunks, ignore_index=True)
            
            # 最终优化
            final_df = self.memory_optimizer.optimize_dataframe(final_df)
            
            # 保存
            output_file = PROCESSED_DATA_DIR / f"ibd_processed_{len(final_df)}.parquet"
            final_df.to_parquet(output_file, index=False)
            
            print(f"\n✅ 数据处理完成!")
            print(f"   文献数量: {len(final_df):,} 篇")
            print(f"   保存格式: Parquet (优化)")
            print(f"   文件大小: {output_file.stat().st_size / 1024**3:.2f} GB")
            
            return output_file
        
        return None

def main():
    parser = argparse.ArgumentParser(description="飞桨平台IBD文献分析系统（增强版）")
    parser.add_argument("--total", type=int, default=600000, 
                       help="目标文献总数")
    parser.add_argument("--batch-size", type=int, default=10000,
                       help="分批大小")
    parser.add_argument("--topics", type=int, default=80,
                       help="主题数量")
    parser.add_argument("--use-gpu", action="store_true", default=True,
                       help="使用GPU加速")
    parser.add_argument("--skip-crawl", action="store_true",
                       help="跳过爬取，使用已有数据")
    parser.add_argument("--optimize-memory", action="store_true", default=True,
                       help="启用内存优化")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("飞桨平台 IBD文献大规模分析系统（增强版）")
    print("=" * 70)
    print(f"🎯 目标文献数: {args.total:,}")
    print(f"📦 分批大小: {args.batch_size}")
    print(f"🧠 主题数量: {args.topics}")
    print(f"⚡ GPU加速: {'是' if args.use_gpu else '否'}")
    print(f"💾 内存优化: {'是' if args.optimize_memory else '否'}")
    print("=" * 70)
    
    # 创建分析器
    analyzer = PaddleIBDAnalyzerEnhanced(use_gpu=args.use_gpu)
    
    try:
        if not args.skip_crawl:
            # 步骤1: 获取数据
            articles = analyzer.crawl_in_batches_optimized(
                query="IBD OR inflammatory bowel disease",
                total_count=args.total,
                batch_size=args.batch_size
            )
            
            if not articles:
                print("❌ 未获取到文献")
                return
        else:
            # 使用已有数据
            print("📁 使用已有数据...")
            raw_files = list(RAW_DATA_DIR.glob("*.json"))
            if not raw_files:
                print("❌ 未找到原始数据文件")
                return
        
        # 步骤2: 处理数据
        processed_file = analyzer.process_large_data_optimized(
            raw_files if args.skip_crawl else None,
            chunk_size=50000
        )
        
        if not processed_file or not processed_file.exists():
            print("❌ 数据处理失败")
            return
        
        # 步骤3: AI分析
        print("\n🧠 开始AI分析阶段...")
        
        # 这里可以添加AI分析代码
        # analyzer.analyze_with_paddle_optimized(processed_file, n_topics=args.topics)
        
    except Exception as e:
        print(f"❌ 程序出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保监控停止
        analyzer.stop_monitoring()
        
        # 显示最终状态
        progress = analyzer.progress_tracker.get_progress()
        print("\n📊 最终状态:")
        for key, value in progress.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 70)
        print("✅ 分析流程完成!")
        print("=" * 70)

if __name__ == "__main__":
    main()