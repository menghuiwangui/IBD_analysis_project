# src/data_collection/data_processor.py
import pandas as pd
import json
import re
from datetime import datetime
from config.settings import *

def load_raw_data(filename):
    """加载原始数据"""
    filepath = RAW_DATA_DIR / filename
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text):
    """清洗文本"""
    if not isinstance(text, str):
        return ""
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符但保留基本标点
    text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
    return text.strip()

def clean_data(articles):
    """清洗数据"""
    df = pd.DataFrame(articles)
    
    if df.empty:
        return df
    
    # 1. 去除重复
    df = df.drop_duplicates(subset=['pmid'], keep='first')
    
    # 2. 处理空值
    df['title'] = df['title'].fillna('')
    df['abstract'] = df['abstract'].fillna('')
    
    # 3. 过滤无摘要的文章
    df = df[df['abstract'].str.strip() != '']
    
    # 4. 清洗文本
    df['title'] = df['title'].apply(clean_text)
    df['abstract'] = df['abstract'].apply(clean_text)
    
    # 5. 创建完整文本
    df['text'] = df['title'] + '. ' + df['abstract']
    
    # 6. 提取年份
    def extract_year(date_str):
        if isinstance(date_str, str):
            match = re.search(r'\d{4}', date_str)
            if match:
                return int(match.group())
        return None
    
    df['year'] = df['pub_date'].apply(extract_year)
    
    # 7. 添加统计信息
    df['title_length'] = df['title'].str.len()
    df['abstract_length'] = df['abstract'].str.len()
    df['text_length'] = df['text'].str.len()
    
    # 8. 重置索引
    df = df.reset_index(drop=True)
    
    return df

def save_processed_data(df, filename="ibd_processed.csv"):
    """保存处理后的数据"""
    filepath = PROCESSED_DATA_DIR / filename
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"✅ Saved processed data to: {filepath}")
    return str(filepath)

def process_data(input_file, output_file="ibd_processed.csv"):
    """完整处理流程"""
    print(f"📥 Loading data from {input_file}...")
    articles = load_raw_data(input_file)
    print(f"   Loaded {len(articles)} raw articles")
    
    print("🧹 Cleaning data...")
    df_clean = clean_data(articles)
    print(f"   After cleaning: {len(df_clean)} articles")
    
    if df_clean.empty:
        print("⚠️  No valid data after cleaning")
        return pd.DataFrame()
    
    print("💾 Saving processed data...")
    save_processed_data(df_clean, output_file)
    
    # 打印统计信息
    print("\n📊 Data Statistics:")
    print(f"   Total articles: {len(df_clean)}")
    print(f"   Average title length: {df_clean['title_length'].mean():.1f} chars")
    print(f"   Average abstract length: {df_clean['abstract_length'].mean():.1f} chars")
    
    if 'year' in df_clean.columns:
        valid_years = df_clean['year'].dropna()
        if not valid_years.empty:
            print(f"   Publication years: {int(valid_years.min())} - {int(valid_years.max())}")
    
    return df_clean