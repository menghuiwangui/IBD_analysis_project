# src/ai_analysis/model_training.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import joblib
import os
import matplotlib.pyplot as plt
from config.settings_paddle import *

class TopicAnalyzer:
    def __init__(self, n_topics=TOPIC_N_COMPONENTS, max_words=MAX_WORDS):
        self.n_topics = n_topics
        self.max_words = max_words
        self.vectorizer = None
        self.lda = None
        self.embedding_model = None
    
    def prepare_texts(self, df, text_column='text'):
        """准备文本数据"""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        texts = df[text_column].fillna('').astype(str).tolist()
        return texts
    
    def fit(self, texts):
        """训练模型"""
        print("🔧 Vectorizing texts...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_words,
            stop_words='english',
            ngram_range=(1, 2)
        )
        X = self.vectorizer.fit_transform(texts)
        
        print(f"🔧 Training LDA with {self.n_topics} topics...")
        self.lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            learning_method='online',
            max_iter=10
        )
        self.lda.fit(X)
        
        return X
    
    def get_topics(self, n_words=10):
        """获取主题"""
        if self.vectorizer is None or self.lda is None:
            raise ValueError("Model not trained yet")
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.lda.components_):
            top_indices = topic.argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append(top_words)
        
        return topics
    
    def predict_topics(self, texts):
        """预测主题"""
        if self.vectorizer is None or self.lda is None:
            raise ValueError("Model not trained yet")
        
        X = self.vectorizer.transform(texts)
        topic_dist = self.lda.transform(X)
        return topic_dist
    
    def generate_embeddings(self, texts):
        """生成嵌入向量"""
        print("🔧 Generating embeddings...")
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def analyze(self, df, save=True):
        """分析数据"""
        # 准备文本
        texts = self.prepare_texts(df)
        print(f"📚 Analyzing {len(texts)} articles...")
        
        # 训练主题模型
        X = self.fit(texts)
        
        # 获取主题
        topics = self.get_topics(n_words=15)
        
        # 主题分布
        topic_dist = self.predict_topics(texts)
        df['dominant_topic'] = topic_dist.argmax(axis=1)
        df['topic_confidence'] = topic_dist.max(axis=1)
        
        # 生成嵌入
        embeddings = self.generate_embeddings(texts)
        
        # 聚类
        n_clusters = min(10, len(texts) // 20)
        if n_clusters > 1:
            print(f"🔧 Clustering into {n_clusters} groups...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(embeddings)
        
        # 保存模型
        if save:
            self.save_model()
            
            # 保存带主题的数据
            output_path = RESULTS_DIR / 'ibd_with_topics.csv'
            df.to_csv(output_path, index=False)
            print(f"✅ Results saved to {output_path}")
            
            # 保存主题
            topics_path = RESULTS_DIR / 'topics.txt'
            with open(topics_path, 'w', encoding='utf-8') as f:
                f.write(f"LDA Topics (n={self.n_topics})\n")
                f.write("=" * 50 + "\n\n")
                for i, topic_words in enumerate(topics):
                    f.write(f"Topic {i}:\n")
                    f.write("  " + ", ".join(topic_words[:10]) + "\n\n")
            
            print(f"✅ Topics saved to {topics_path}")
        
        return df, topics
    
    def save_model(self):
        """保存模型"""
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, RESULTS_DIR / 'vectorizer.pkl')
        if self.lda is not None:
            joblib.dump(self.lda, RESULTS_DIR / 'lda_model.pkl')
        print("✅ Models saved")
    
    def load_model(self):
        """加载模型"""
        self.vectorizer = joblib.load(RESULTS_DIR / 'vectorizer.pkl')
        self.lda = joblib.load(RESULTS_DIR / 'lda_model.pkl')
        print("✅ Models loaded")