# src/data_collection/data_processor_enhanced.py
"""
增强版数据处理模块，支持大规模数据处理
"""
import pandas as pd
import numpy as np
import json
import re
import html
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from datetime import datetime
import hashlib
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from config.settings_paddle import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDataProcessor:
    """增强版数据处理器，支持并行处理和大数据优化"""
    
    def __init__(self, n_workers: int = None):
        """
        初始化处理器
        
        Args:
            n_workers: 并行工作进程数，None为自动检测
        """
        self.n_workers = n_workers or min(mp.cpu_count(), 8)
        logger.info(f"初始化数据处理器，使用 {self.n_workers} 个工作进程")
        
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        if not isinstance(text, str):
            return ""
        
        # 解码HTML实体
        text = html.unescape(text)
        
        # 移除特殊字符但保留基本标点
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        
        # 规范化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除换行符和多余空格
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = text.strip()
        
        return text
    
    def extract_year(self, date_str: str) -> Optional[int]:
        """从日期字符串中提取年份"""
        if not isinstance(date_str, str):
            return None
        
        # 匹配年份模式
        year_patterns = [
            r'(\d{4})',  # YYYY
            r'(\d{4})-',  # YYYY-
            r'(\d{4})/',  # YYYY/
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    year = int(match.group(1))
                    if 1900 <= year <= datetime.now().year:
                        return year
                except:
                    continue
        
        return None
    
    def extract_doi(self, article: Dict) -> Optional[str]:
        """提取DOI"""
        # 从PMID生成DOI
        pmid = article.get('pmid', '')
        if pmid:
            return f"10.0000/pubmed.{pmid}"
        return None
    
    def calculate_text_hash(self, text: str) -> str:
        """计算文本哈希值用于去重"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def process_article_batch(self, articles_batch: List[Dict]) -> pd.DataFrame:
        """处理一批文章"""
        processed = []
        
        for article in articles_batch:
            try:
                # 基础字段
                pmid = article.get('pmid', '')
                title = self.clean_text(article.get('title', ''))
                abstract = self.clean_text(article.get('abstract', ''))
                
                # 跳过无标题或无摘要的文章
                if not title or len(title) < 10 or not abstract or len(abstract) < 50:
                    continue
                
                # 作者处理
                authors_raw = article.get('authors', '')
                if isinstance(authors_raw, str):
                    authors = [a.strip() for a in authors_raw.split(';') if a.strip()]
                else:
                    authors = []
                
                # 年份提取
                pub_date = article.get('pub_date', '')
                year = self.extract_year(pub_date)
                
                # 期刊信息
                journal = article.get('journal', '')
                
                # MeSH术语
                mesh_terms_raw = article.get('mesh_terms', '')
                if isinstance(mesh_terms_raw, str):
                    mesh_terms = [t.strip() for t in mesh_terms_raw.split(';') if t.strip()]
                else:
                    mesh_terms = []
                
                # 文章类型
                pub_types_raw = article.get('publication_types', '')
                if isinstance(pub_types_raw, str):
                    pub_types = [t.strip() for t in pub_types_raw.split(';') if t.strip()]
                else:
                    pub_types = []
                
                # 构建完整文本
                full_text = f"{title}. {abstract}"
                
                # 计算哈希用于去重
                text_hash = self.calculate_text_hash(full_text)
                
                processed.append({
                    'pmid': pmid,
                    'doi': self.extract_doi(article),
                    'title': title,
                    'abstract': abstract,
                    'authors': '; '.join(authors[:10]),  # 最多保留10位作者
                    'journal': journal,
                    'pub_date': pub_date,
                    'year': year,
                    'mesh_terms': '; '.join(mesh_terms[:20]),  # 最多保留20个MeSH术语
                    'publication_types': '; '.join(pub_types),
                    'full_text': full_text,
                    'text_hash': text_hash,
                    'title_length': len(title),
                    'abstract_length': len(abstract),
                    'full_text_length': len(full_text),
                    'author_count': len(authors),
                    'mesh_term_count': len(mesh_terms),
                    'fetch_time': article.get('fetch_time', '')
                })
                
            except Exception as e:
                logger.warning(f"处理文章出错 (PMID: {article.get('pmid', 'unknown')}): {e}")
                continue
        
        return pd.DataFrame(processed)
    
    def clean_data_parallel(self, articles: List[Dict], chunk_size: int = 1000) -> pd.DataFrame:
        """并行清洗数据"""
        logger.info(f"开始并行处理 {len(articles)} 篇文章，分块大小: {chunk_size}")
        
        # 分块
        chunks = [articles[i:i + chunk_size] for i in range(0, len(articles), chunk_size)]
        
        # 并行处理
        with mp.Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(self.process_article_batch, chunks),
                total=len(chunks),
                desc="并行处理文章"
            ))
        
        # 合并结果
        if results:
            df = pd.concat(results, ignore_index=True)
            logger.info(f"并行处理完成，得到 {len(df)} 篇文章")
            return df
        else:
            logger.warning("并行处理无结果")
            return pd.DataFrame()
    
    def deduplicate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """去重DataFrame"""
        if df.empty:
            return df
        
        # 记录原始数量
        original_count = len(df)
        
        # 1. 基于text_hash去重
        df = df.drop_duplicates(subset=['text_hash'], keep='first')
        
        # 2. 基于标题相似度去重（简单的）
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        # 3. 移除太短的文章
        df = df[df['full_text_length'] >= 100]
        
        # 4. 移除无年份的文章
        df = df[df['year'].notna()]
        
        # 记录去重后数量
        final_count = len(df)
        removed_count = original_count - final_count
        
        logger.info(f"去重完成: 移除 {removed_count} 篇重复/无效文章，剩余 {final_count} 篇")
        
        return df.reset_index(drop=True)
    
    def extract_keywords_from_text(self, text: str, top_n: int = 10) -> List[str]:
        """从文本中提取关键词（简单TF-IDF实现）"""
        if not text or len(text) < 50:
            return []
        
        # 简单的关键词提取（实际中应该用更复杂的方法）
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # 移除停用词
        stopwords = set([
            'this', 'that', 'with', 'from', 'have', 'were', 'which',
            'study', 'studies', 'research', 'results', 'conclusion',
            'inflammatory', 'bowel', 'disease', 'ibd', 'patients'
        ])
        
        filtered_words = [w for w in words if w not in stopwords]
        
        # 统计词频
        from collections import Counter
        word_counts = Counter(filtered_words)
        
        # 返回最常见的词
        return [word for word, _ in word_counts.most_common(top_n)]
    
    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """丰富数据框，添加额外特征"""
        if df.empty:
            return df
        
        df_enriched = df.copy()
        
        # 添加关键词
        logger.info("提取关键词...")
        df_enriched['keywords'] = df_enriched['full_text'].apply(
            lambda x: '; '.join(self.extract_keywords_from_text(x, top_n=15))
        )
        
        # 添加疾病分类
        logger.info("分类疾病类型...")
        def classify_disease(text):
            text_lower = text.lower()
            if 'crohn' in text_lower:
                return 'Crohn\'s Disease'
            elif 'ulcerative colitis' in text_lower:
                return 'Ulcerative Colitis'
            elif 'ibd' in text_lower or 'inflammatory bowel disease' in text_lower:
                return 'IBD General'
            else:
                return 'Other'
        
        df_enriched['disease_type'] = df_enriched['full_text'].apply(classify_disease)
        
        # 添加研究类型分类
        logger.info("分类研究类型...")
        def classify_study_type(text):
            text_lower = text.lower()
            if any(word in text_lower for word in ['randomized', 'trial', 'clinical trial']):
                return 'Clinical Trial'
            elif any(word in text_lower for word in ['meta-analysis', 'systematic review']):
                return 'Meta-Analysis/Review'
            elif any(word in text_lower for word in ['genetic', 'genome', 'gene', 'snp']):
                return 'Genetic Study'
            elif any(word in text_lower for word in ['microbiome', 'microbiota', 'bacteria']):
                return 'Microbiome Study'
            elif any(word in text_lower for word in ['immun', 'cytokine', 't cell']):
                return 'Immunology Study'
            else:
                return 'Other Research'
        
        df_enriched['study_type'] = df_enriched['full_text'].apply(classify_study_type)
        
        # 计算文本复杂度
        logger.info("计算文本特征...")
        df_enriched['avg_word_length'] = df_enriched['full_text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x else 0
        )
        
        df_enriched['unique_word_ratio'] = df_enriched['full_text'].apply(
            lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0
        )
        
        # 添加处理时间戳
        df_enriched['process_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"数据丰富完成，添加了 {len(df_enriched.columns) - len(df.columns)} 个新特征")
        
        return df_enriched
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = None) -> Path:
        """保存处理后的数据"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ibd_processed_{len(df)}_{timestamp}"
        
        # 保存为Parquet（更高效）
        parquet_path = PROCESSED_DATA_DIR / f"{filename}.parquet"
        df.to_parquet(parquet_path, index=False, compression='gzip')
        
        # 同时保存CSV用于查看
        csv_path = PROCESSED_DATA_DIR / f"{filename}.csv"
        # 只保存前10000行到CSV
        df.head(10000).to_csv(csv_path, index=False, encoding='utf-8')
        
        # 保存统计信息
        stats_path = PROCESSED_DATA_DIR / f"{filename}_stats.json"
        stats = self.get_data_statistics(df)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据已保存:")
        logger.info(f"  - Parquet: {parquet_path} ({parquet_path.stat().st_size / 1024**3:.2f} GB)")
        logger.info(f"  - CSV (样本): {csv_path}")
        logger.info(f"  - 统计信息: {stats_path}")
        
        return parquet_path
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据统计信息"""
        if df.empty:
            return {}
        
        stats = {
            'total_articles': len(df),
            'year_range': {
                'min': int(df['year'].min()) if df['year'].notna().any() else None,
                'max': int(df['year'].max()) if df['year'].notna().any() else None,
                'avg': float(df['year'].mean()) if df['year'].notna().any() else None
            },
            'text_statistics': {
                'avg_title_length': float(df['title_length'].mean()),
                'avg_abstract_length': float(df['abstract_length'].mean()),
                'avg_full_text_length': float(df['full_text_length'].mean())
            },
            'disease_distribution': df['disease_type'].value_counts().to_dict(),
            'study_type_distribution': df['study_type'].value_counts().to_dict(),
            'journal_distribution': df['journal'].value_counts().head(20).to_dict(),
            'author_statistics': {
                'avg_authors_per_article': float(df['author_count'].mean()),
                'max_authors': int(df['author_count'].max())
            },
            'publication_years': {
                str(year): int(count) for year, count in df['year'].value_counts().sort_index().items()
            }
        }
        
        return stats
    
    def process_large_dataset(self, input_files: List[Path], output_name: str = None) -> Path:
        """处理大规模数据集的主函数"""
        logger.info(f"开始处理大规模数据集，输入文件: {len(input_files)} 个")
        
        all_articles = []
        
        # 1. 加载所有数据
        for input_file in tqdm(input_files, desc="加载数据文件"):
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_articles.extend(data)
                    logger.info(f"  加载 {len(data)} 篇文章来自 {input_file.name}")
            except Exception as e:
                logger.error(f"加载文件失败 {input_file}: {e}")
        
        if not all_articles:
            logger.error("未加载到任何文章数据")
            return None
        
        logger.info(f"总共加载 {len(all_articles)} 篇文章")
        
        # 2. 并行处理
        logger.info("开始并行数据清洗...")
        df = self.clean_data_parallel(all_articles)
        
        if df.empty:
            logger.error("数据处理后无有效文章")
            return None
        
        # 3. 去重
        logger.info("开始数据去重...")
        df_dedup = self.deduplicate_dataframe(df)
        
        # 4. 数据丰富
        logger.info("开始数据丰富...")
        df_enriched = self.enrich_dataframe(df_dedup)
        
        # 5. 保存
        logger.info("保存处理后的数据...")
        output_path = self.save_processed_data(df_enriched, output_name)
        
        # 6. 打印统计信息
        stats = self.get_data_statistics(df_enriched)
        logger.info(f"\n📊 数据处理统计:")
        logger.info(f"   总文章数: {stats['total_articles']:,}")
        logger.info(f"   年份范围: {stats['year_range']['min']} - {stats['year_range']['max']}")
        logger.info(f"   疾病分布: {', '.join([f'{k}: {v}' for k, v in stats['disease_distribution'].items()])}")
        logger.info(f"   研究类型: {', '.join([f'{k}: {v}' for k, v in stats['study_type_distribution'].items()])}")
        
        return output_path

# 简化的处理函数（兼容旧接口）
def clean_data(articles: List[Dict]) -> pd.DataFrame:
    """简化接口，用于兼容旧代码"""
    processor = EnhancedDataProcessor(n_workers=4)
    return processor.clean_data_parallel(articles)

def process_data(input_file: Path, output_file: str = "ibd_processed.csv") -> pd.DataFrame:
    """简化接口，用于兼容旧代码"""
    processor = EnhancedDataProcessor(n_workers=4)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    df = processor.clean_data_parallel(articles)
    df = processor.deduplicate_dataframe(df)
    df = processor.enrich_dataframe(df)
    
    output_path = PROCESSED_DATA_DIR / output_file
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    return df