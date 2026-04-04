# src/ai_analysis/topic_model_paddle.py
"""
基于PaddlePaddle的主题分析模型
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# 尝试导入PaddlePaddle，如果失败则回退到CPU
try:
    import paddle
    import paddle.nn as nn
    import paddle.nn.functional as F
    from paddlenlp.transformers import BertModel, BertTokenizer
    PADDLE_AVAILABLE = True
    print("✅ PaddlePaddle 可用，将使用GPU加速")
except ImportError:
    PADDLE_AVAILABLE = False
    print("⚠️ PaddlePaddle 不可用，将使用CPU版本")

# 导入其他必要的库
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE, UMAP
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import json
from tqdm import tqdm
from config.settings_paddle import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaddleTopicAnalyzer:
    """基于PaddlePaddle的主题分析器，支持GPU加速"""
    
    def __init__(self, n_topics: int = TOPIC_N_COMPONENTS, 
                 max_words: int = MAX_WORDS,
                 use_gpu: bool = True,
                 embedding_model: str = None):
        """
        初始化主题分析器
        
        Args:
            n_topics: 主题数量
            max_words: 最大词汇量
            use_gpu: 是否使用GPU
            embedding_model: 嵌入模型名称
        """
        self.n_topics = n_topics
        self.max_words = max_words
        self.use_gpu = use_gpu and PADDLE_AVAILABLE
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL
        
        # 初始化模型
        self.vectorizer = None
        self.lda_model = None
        self.nmf_model = None
        self.kmeans = None
        self.embedding_model = None
        self.paddle_model = None
        
        # 结果存储
        self.topics = None
        self.topic_distributions = None
        self.embeddings = None
        
        logger.info(f"初始化主题分析器: topics={n_topics}, use_gpu={use_gpu}")
        
        # 设置GPU
        if self.use_gpu:
            try:
                paddle.set_device('gpu')
                logger.info("✅ 已设置GPU设备")
            except:
                logger.warning("⚠️ GPU不可用，将使用CPU")
                self.use_gpu = False
                paddle.set_device('cpu')
    
    def load_embedding_model(self):
        """加载嵌入模型"""
        logger.info(f"加载嵌入模型: {self.embedding_model_name}")
        
        if self.use_gpu and 'ernie' in self.embedding_model_name.lower():
            # 尝试加载ERNIE模型（中文更好）
            try:
                from paddlenlp.transformers import AutoModel, AutoTokenizer
                self.paddle_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
                self.paddle_model = AutoModel.from_pretrained(self.embedding_model_name)
                logger.info(f"✅ 加载PaddleNLP模型: {self.embedding_model_name}")
            except:
                logger.warning("⚠️ 无法加载PaddleNLP模型，回退到sentence-transformers")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
        else:
            # 使用sentence-transformers
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        logger.info("✅ 嵌入模型加载完成")
    
    def create_embeddings_paddle(self, texts: List[str]) -> np.ndarray:
        """使用PaddlePaddle创建嵌入"""
        if not self.paddle_model:
            self.load_embedding_model()
        
        if self.paddle_model:
            logger.info("使用PaddlePaddle生成嵌入...")
            embeddings = []
            
            batch_size = 32
            for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入"):
                batch_texts = texts[i:i + batch_size]
                
                # 分词
                inputs = self.paddle_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pd'
                )
                
                # 前向传播
                with paddle.no_grad():
                    outputs = self.paddle_model(**inputs)
                    # 使用[CLS] token作为句子表示
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                
                embeddings.append(batch_embeddings)
            
            return np.vstack(embeddings)
        else:
            # 回退到sentence-transformers
            return self.create_embeddings(texts)
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """创建文本嵌入"""
        if not self.embedding_model:
            self.load_embedding_model()
        
        logger.info(f"为 {len(texts)} 个文本生成嵌入...")
        
        if self.paddle_model:
            return self.create_embeddings_paddle(texts)
        else:
            # 使用sentence-transformers
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
            return embeddings
    
    def prepare_texts(self, df: pd.DataFrame, text_column: str = 'full_text') -> List[str]:
        """准备文本数据"""
        if text_column not in df.columns:
            raise ValueError(f"文本列 '{text_column}' 不存在")
        
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # 过滤太短的文本
        texts = [text for text in texts if len(text) >= 50]
        
        logger.info(f"准备 {len(texts)} 个文本进行分析")
        return texts
    
    def create_tfidf_features(self, texts: List[str]) -> Tuple[Any, np.ndarray]:
        """创建TF-IDF特征"""
        logger.info("创建TF-IDF特征矩阵...")
        
        # 使用TF-IDF向量化
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_words,
            stop_words='english',
            ngram_range=(1, 2),  # 包括unigram和bigram
            min_df=5,  # 至少出现在5个文档中
            max_df=0.8  # 最多出现在80%的文档中
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF矩阵形状: {tfidf_matrix.shape}")
        
        return tfidf_matrix
    
    def train_topic_model(self, tfidf_matrix, method: str = 'lda') -> Any:
        """训练主题模型"""
        logger.info(f"训练主题模型 (方法: {method}, 主题数: {self.n_topics})...")
        
        if method == 'lda':
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                learning_method='online',  # 在线学习，适合大数据
                max_iter=20,  # 增加迭代次数
                batch_size=128,  # 批处理大小
                evaluate_every=5,  # 每5个批次评估一次
                n_jobs=-1  # 使用所有CPU核心
            )
            self.lda_model.fit(tfidf_matrix)
            model = self.lda_model
            
        elif method == 'nmf':
            self.nmf_model = NMF(
                n_components=self.n_topics,
                random_state=42,
                max_iter=200,
                init='nndsvda'  # 更好的初始化
            )
            self.nmf_model.fit(tfidf_matrix)
            model = self.nmf_model
        
        else:
            raise ValueError(f"不支持的模型方法: {method}")
        
        logger.info("✅ 主题模型训练完成")
        return model
    
    def get_topic_keywords(self, n_words: int = 15) -> List[List[str]]:
        """获取主题关键词"""
        if not self.vectorizer or not (self.lda_model or self.nmf_model):
            raise ValueError("请先训练模型")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        if self.lda_model:
            model = self.lda_model
        else:
            model = self.nmf_model
        
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            # 获取权重最高的词
            top_indices = topic.argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_indices]
            
            # 计算权重
            weights = topic[top_indices]
            
            # 标准化权重
            weights = weights / weights.sum()
            
            topics.append({
                'topic_id': topic_idx,
                'keywords': top_words,
                'weights': weights.tolist(),
                'top_keywords': ', '.join(top_words[:5])
            })
        
        return topics
    
    def predict_topic_distributions(self, tfidf_matrix) -> np.ndarray:
        """预测主题分布"""
        if self.lda_model:
            return self.lda_model.transform(tfidf_matrix)
        elif self.nmf_model:
            return self.nmf_model.transform(tfidf_matrix)
        else:
            raise ValueError("没有可用的主题模型")
    
    def cluster_embeddings(self, embeddings: np.ndarray, method: str = 'kmeans') -> np.ndarray:
        """对嵌入进行聚类"""
        logger.info(f"对嵌入进行聚类 (方法: {method})...")
        
        if method == 'kmeans':
            # 自动确定聚类数
            n_clusters = min(self.n_topics * 2, len(embeddings) // 10)
            n_clusters = max(2, n_clusters)  # 至少2个聚类
            
            self.kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            clusters = self.kmeans.fit_predict(embeddings)
            
            # 计算轮廓系数
            if n_clusters > 1:
                silhouette_avg = silhouette_score(embeddings, clusters)
                logger.info(f"KMeans轮廓系数: {silhouette_avg:.3f}")
            
        elif method == 'dbscan':
            self.dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = self.dbscan.fit_predict(embeddings)
            
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            logger.info(f"DBSCAN找到 {n_clusters} 个聚类")
            
        else:
            raise ValueError(f"不支持的聚类方法: {method}")
        
        return clusters
    
    def reduce_dimensions(self, embeddings: np.ndarray, method: str = 'umap') -> np.ndarray:
        """降维用于可视化"""
        logger.info(f"降维 (方法: {method})...")
        
        if method == 'umap':
            reducer = UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine'
            )
        elif method == 'tsne':
            reducer = TSNE(
                n_components=2,
                random_state=42,
                perplexity=30,
                n_iter=1000
            )
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        return embeddings_2d
    
    def analyze_topics_advanced(self, df: pd.DataFrame, 
                                text_column: str = 'full_text',
                                save_results: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
        """高级主题分析"""
        start_time = time.time()
        
        # 1. 准备文本
        texts = self.prepare_texts(df, text_column)
        if len(texts) < self.n_topics:
            logger.warning(f"文本数量 ({len(texts)}) 少于主题数 ({self.n_topics})，减少主题数")
            self.n_topics = max(2, len(texts) // 10)
        
        # 2. 创建TF-IDF特征
        tfidf_matrix = self.create_tfidf_features(texts)
        
        # 3. 训练主题模型
        self.train_topic_model(tfidf_matrix, method='lda')
        
        # 4. 获取主题
        topics = self.get_topic_keywords(n_words=20)
        
        # 5. 预测主题分布
        topic_dist = self.predict_topic_distributions(tfidf_matrix)
        df_topic = df.copy().iloc[:len(texts)]  # 只保留有文本的行
        
        # 添加主题信息
        df_topic['dominant_topic'] = topic_dist.argmax(axis=1)
        df_topic['topic_confidence'] = topic_dist.max(axis=1)
        
        # 添加每个主题的权重
        for i in range(self.n_topics):
            df_topic[f'topic_{i}_weight'] = topic_dist[:, i]
        
        # 6. 创建嵌入
        logger.info("创建文本嵌入...")
        self.embeddings = self.create_embeddings(texts)
        
        # 7. 聚类
        logger.info("对嵌入进行聚类...")
        df_topic['cluster'] = self.cluster_embeddings(self.embeddings, method='kmeans')
        
        # 8. 降维用于可视化
        logger.info("降维用于可视化...")
        embeddings_2d = self.reduce_dimensions(self.embeddings, method='umap')
        df_topic['x'] = embeddings_2d[:, 0]
        df_topic['y'] = embeddings_2d[:, 1]
        
        # 9. 保存结果
        if save_results:
            self.save_results(df_topic, topics)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ 主题分析完成，耗时: {elapsed_time:.1f} 秒")
        
        return df_topic, topics
    
    def save_results(self, df: pd.DataFrame, topics: List[Dict]):
        """保存分析结果"""
        logger.info("保存分析结果...")
        
        # 保存DataFrame
        output_path = RESULTS_DIR / 'ibd_with_topics_detailed.parquet'
        df.to_parquet(output_path, index=False)
        logger.info(f"✅ 数据已保存: {output_path}")
        
        # 保存主题信息
        topics_path = RESULTS_DIR / 'topics_detailed.json'
        with open(topics_path, 'w', encoding='utf-8') as f:
            json.dump(topics, f, indent=2, ensure_ascii=False)
        
        # 保存主题关键词文本文件
        topics_text_path = RESULTS_DIR / 'topics_keywords.txt'
        with open(topics_text_path, 'w', encoding='utf-8') as f:
            f.write(f"IBD文献主题分析结果 (共{len(topics)}个主题)\n")
            f.write("=" * 80 + "\n\n")
            
            for topic in topics:
                f.write(f"主题 {topic['topic_id']}:\n")
                f.write(f"  关键词: {topic['top_keywords']}\n")
                f.write(f"  完整关键词: {', '.join(topic['keywords'])}\n\n")
        
        # 保存模型
        model_dir = MODEL_DIR
        model_dir.mkdir(exist_ok=True)
        
        if self.vectorizer:
            joblib.dump(self.vectorizer, model_dir / 'vectorizer.pkl')
        if self.lda_model:
            joblib.dump(self.lda_model, model_dir / 'lda_model.pkl')
        if self.kmeans:
            joblib.dump(self.kmeans, model_dir / 'kmeans_model.pkl')
        
        logger.info("✅ 模型已保存")
        
        # 生成可视化
        self.create_visualizations(df, topics)
    
    def create_visualizations(self, df: pd.DataFrame, topics: List[Dict]):
        """创建可视化"""
        logger.info("创建可视化...")
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            
            # 1. 主题分布图
            plt.figure(figsize=(12, 6))
            topic_counts = df['dominant_topic'].value_counts().sort_index()
            plt.bar(topic_counts.index, topic_counts.values)
            plt.xlabel('Topic ID')
            plt.ylabel('Number of Articles')
            plt.title('Topic Distribution')
            plt.savefig(RESULTS_DIR / 'topic_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 主题置信度分布
            plt.figure(figsize=(10, 6))
            plt.hist(df['topic_confidence'], bins=50, alpha=0.7, color='blue')
            plt.xlabel('Topic Confidence')
            plt.ylabel('Frequency')
            plt.title('Distribution of Topic Confidence Scores')
            plt.savefig(RESULTS_DIR / 'topic_confidence_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 主题聚类可视化
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(df['x'], df['y'], 
                                 c=df['dominant_topic'], 
                                 cmap='tab20', 
                                 s=10, 
                                 alpha=0.6)
            plt.colorbar(scatter, label='Topic')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.title('Topic Clustering Visualization')
            plt.savefig(RESULTS_DIR / 'topic_clustering.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. 词云（每个主题）
            for topic in topics[:10]:  # 只保存前10个主题的词云
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    max_words=50
                ).generate_from_frequencies(
                    dict(zip(topic['keywords'], topic['weights']))
                )
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Topic {topic["topic_id"]} - Word Cloud')
                plt.savefig(RESULTS_DIR / f'topic_{topic["topic_id"]}_wordcloud.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info("✅ 可视化已生成")
            
        except Exception as e:
            logger.warning(f"可视化生成失败: {e}")

def analyze_topics(df: pd.DataFrame, n_topics: int = 10, use_gpu: bool = True) -> pd.DataFrame:
    """简化接口，用于兼容旧代码"""
    analyzer = PaddleTopicAnalyzer(n_topics=n_topics, use_gpu=use_gpu)
    df_with_topics, topics = analyzer.analyze_topics_advanced(df, save_results=True)
    return df_with_topics