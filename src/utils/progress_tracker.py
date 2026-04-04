# src/utils/progress_tracker.py
"""
进度跟踪和断点续传
"""
import os
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
import hashlib
from typing import Dict, Any, Optional
import pandas as pd

class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, checkpoint_dir: Path, task_name: str = "ibd_analysis"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.task_name = task_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 状态文件路径
        self.state_file = self.checkpoint_dir / f"{task_name}_state.json"
        self.checkpoint_files = {}
        
    def save_state(self, state: Dict[str, Any]):
        """保存状态"""
        state['timestamp'] = datetime.now().isoformat()
        state['task_name'] = self.task_name
        
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 状态已保存: {self.state_file}")
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """加载状态"""
        if not self.state_file.exists():
            return None
        
        with open(self.state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_checkpoint(self, data: Any, checkpoint_id: str, 
                       checkpoint_type: str = "data"):
        """保存检查点"""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_type}_{checkpoint_id}.pkl"
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.checkpoint_files[checkpoint_id] = checkpoint_file
        print(f"💾 检查点已保存: {checkpoint_file}")
        
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_id: str, checkpoint_type: str = "data") -> Any:
        """加载检查点"""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_type}_{checkpoint_id}.pkl"
        
        if not checkpoint_file.exists():
            return None
        
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    
    def save_dataframe_checkpoint(self, df: pd.DataFrame, checkpoint_id: str, 
                                 format: str = "parquet"):
        """保存DataFrame检查点"""
        if format == "parquet":
            filepath = self.checkpoint_dir / f"df_{checkpoint_id}.parquet"
            df.to_parquet(filepath, index=False)
        else:
            filepath = self.checkpoint_dir / f"df_{checkpoint_id}.csv"
            df.to_csv(filepath, index=False)
        
        print(f"💾 DataFrame检查点已保存: {filepath}")
        return filepath
    
    def load_dataframe_checkpoint(self, checkpoint_id: str, 
                                 format: str = "parquet") -> pd.DataFrame:
        """加载DataFrame检查点"""
        if format == "parquet":
            filepath = self.checkpoint_dir / f"df_{checkpoint_id}.parquet"
            if filepath.exists():
                return pd.read_parquet(filepath)
        else:
            filepath = self.checkpoint_dir / f"df_{checkpoint_id}.csv"
            if filepath.exists():
                return pd.read_csv(filepath)
        
        return None
    
    def get_progress(self) -> Dict[str, Any]:
        """获取进度信息"""
        state = self.load_state()
        if not state:
            return {"status": "not_started"}
        
        # 计算进度
        total = state.get("total", 0)
        current = state.get("current", 0)
        
        if total > 0:
            percentage = (current / total) * 100
        else:
            percentage = 0
        
        return {
            "task": self.task_name,
            "status": state.get("status", "unknown"),
            "progress": f"{current}/{total}",
            "percentage": f"{percentage:.1f}%",
            "timestamp": state.get("timestamp", ""),
            "estimated_time_remaining": self.estimate_time_remaining(state)
        }
    
    def estimate_time_remaining(self, state: Dict[str, Any]) -> str:
        """估计剩余时间"""
        if "start_time" not in state or "current" not in state or "total" not in state:
            return "未知"
        
        start_time = datetime.fromisoformat(state["start_time"])
        current_time = datetime.now()
        elapsed = (current_time - start_time).total_seconds()
        
        current = state["current"]
        total = state["total"]
        
        if current == 0:
            return "未知"
        
        avg_time_per_item = elapsed / current
        remaining_items = total - current
        remaining_seconds = avg_time_per_item * remaining_items
        
        if remaining_seconds < 60:
            return f"{int(remaining_seconds)}秒"
        elif remaining_seconds < 3600:
            return f"{int(remaining_seconds/60)}分钟"
        else:
            return f"{int(remaining_seconds/3600)}小时"
    
    def log_step(self, step_name: str, details: Dict[str, Any] = None):
        """记录步骤"""
        log_file = self.checkpoint_dir / f"{self.task_name}_steps.log"
        
        log_entry = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")