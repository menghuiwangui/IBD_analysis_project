# src/utils/memory_optimizer.py
"""
内存优化工具
"""
import pandas as pd
import numpy as np
import gc
import psutil
import os
from typing import Generator, List, Any
import warnings
warnings.filterwarnings('ignore')

class MemoryOptimizer:
    """内存优化工具类"""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame内存使用"""
        df_optimized = df.copy()
        
        # 优化数值类型
        for col in df_optimized.select_dtypes(include=['int64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        
        for col in df_optimized.select_dtypes(include=['float64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # 优化字符串类型
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if df_optimized[col].nunique() / len(df_optimized) < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
        
        return df_optimized
    
    @staticmethod
    def process_in_chunks(df: pd.DataFrame, chunk_size: int = 10000) -> Generator[pd.DataFrame, None, None]:
        """分批处理DataFrame"""
        total_rows = len(df)
        
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = df.iloc[start:end].copy()
            
            yield chunk
            
            # 手动垃圾回收
            del chunk
            gc.collect()
    
    @staticmethod
    def batch_process(func, data: List[Any], batch_size: int = 1000, **kwargs):
        """批量处理函数"""
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_results = func(batch, **kwargs)
            results.extend(batch_results)
            
            # 清理内存
            del batch
            del batch_results
            gc.collect()
            
            if (i // batch_size) % 10 == 0:
                MemoryOptimizer.print_memory_usage(f"批次 {i//batch_size}")
        
        return results
    
    @staticmethod
    def save_large_object(obj: Any, filepath: str, use_pickle: bool = True):
        """保存大对象（支持分块）"""
        if use_pickle:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            import joblib
            joblib.dump(obj, filepath, compress=3)
    
    @staticmethod
    def load_large_object(filepath: str, use_pickle: bool = True) -> Any:
        """加载大对象"""
        if use_pickle:
            import pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            import joblib
            return joblib.load(filepath)
    
    @staticmethod
    def print_memory_usage(label: str = ""):
        """打印内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        print(f"{label} - 内存使用: {memory_info.rss / 1024**3:.2f} GB")
    
    @staticmethod
    def clear_memory():
        """清理内存"""
        gc.collect()
        
        # 尝试清理TensorFlow/PyTorch缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except:
            pass