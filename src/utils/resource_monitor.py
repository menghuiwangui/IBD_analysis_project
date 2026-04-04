# src/utils/resource_monitor.py
"""
资源监控
"""
import psutil
import time
import threading
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, log_dir: Path, interval: int = 60):
        """
        初始化资源监控器
        
        Args:
            log_dir: 日志目录
            interval: 监控间隔（秒）
        """
        self.log_dir = Path(log_dir)
        self.interval = interval
        self.monitoring = False
        self.thread = None
        
        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件
        self.log_file = self.log_dir / f"resource_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    def start(self):
        """开始监控"""
        if self.monitoring:
            print("⚠️ 监控已经在运行")
            return
        
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"✅ 资源监控已启动，日志: {self.log_file}")
    
    def stop(self):
        """停止监控"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=self.interval * 2)
        print("⏹️ 资源监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            while self.monitoring:
                try:
                    # 获取系统资源
                    resources = self._get_system_resources()
                    
                    # 写入日志
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "resources": resources
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    f.flush()
                    
                    # 打印到控制台
                    self._print_resources(resources)
                    
                except Exception as e:
                    print(f"监控错误: {e}")
                
                # 等待
                for _ in range(self.interval):
                    if not self.monitoring:
                        break
                    time.sleep(1)
    
    def _get_system_resources(self) -> dict:
        """获取系统资源信息"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # 内存
        memory = psutil.virtual_memory()
        
        # 磁盘
        disk = psutil.disk_usage('/')
        
        # 网络（可选）
        try:
            net_io = psutil.net_io_counters()
            network = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv
            }
        except:
            network = None
        
        # 进程特定（当前Python进程）
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count
            },
            "memory": {
                "total_gb": memory.total / 1024**3,
                "available_gb": memory.available / 1024**3,
                "used_gb": memory.used / 1024**3,
                "percent": memory.percent
            },
            "process_memory": {
                "rss_gb": process_memory.rss / 1024**3,  # 驻留集大小
                "vms_gb": process_memory.vms / 1024**3   # 虚拟内存大小
            },
            "disk": {
                "total_gb": disk.total / 1024**3,
                "used_gb": disk.used / 1024**3,
                "free_gb": disk.free / 1024**3,
                "percent": disk.percent
            },
            "network": network
        }
    
    def _print_resources(self, resources: dict):
        """打印资源信息"""
        cpu = resources['cpu']
        memory = resources['memory']
        process_mem = resources['process_memory']
        disk = resources['disk']
        
        print(f"💻 CPU: {cpu['percent']:.1f}% | "
              f"🧠 内存: {memory['percent']:.1f}% ({process_mem['rss_gb']:.1f}GB) | "
              f"💾 磁盘: {disk['percent']:.1f}%")
    
    def generate_report(self) -> dict:
        """生成监控报告"""
        if not self.log_file.exists():
            return {"error": "日志文件不存在"}
        
        data_points = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_points.append(json.loads(line.strip()))
        
        if not data_points:
            return {"error": "无监控数据"}
        
        # 计算统计信息
        cpu_values = [p['resources']['cpu']['percent'] for p in data_points]
        memory_values = [p['resources']['memory']['percent'] for p in data_points]
        process_memory_values = [p['resources']['process_memory']['rss_gb'] for p in data_points]
        
        return {
            "summary": {
                "monitoring_period": {
                    "start": data_points[0]['timestamp'],
                    "end": data_points[-1]['timestamp'],
                    "total_points": len(data_points)
                },
                "cpu": {
                    "average": sum(cpu_values) / len(cpu_values),
                    "max": max(cpu_values),
                    "min": min(cpu_values)
                },
                "memory": {
                    "average": sum(memory_values) / len(memory_values),
                    "max": max(memory_values),
                    "min": min(memory_values)
                },
                "process_memory": {
                    "average_gb": sum(process_memory_values) / len(process_memory_values),
                    "max_gb": max(process_memory_values),
                    "min_gb": min(process_memory_values)
                }
            },
            "recommendations": self._generate_recommendations(data_points)
        }
    
    def _generate_recommendations(self, data_points: list) -> list:
        """生成优化建议"""
        recommendations = []
        
        # 分析内存使用
        memory_values = [p['resources']['memory']['percent'] for p in data_points]
        avg_memory = sum(memory_values) / len(memory_values)
        
        if avg_memory > 80:
            recommendations.append("⚠️ 内存使用过高，考虑增加内存或优化代码")
        elif avg_memory > 60:
            recommendations.append("ℹ️ 内存使用较高，建议监控")
        
        # 分析CPU使用
        cpu_values = [p['resources']['cpu']['percent'] for p in data_points]
        avg_cpu = sum(cpu_values) / len(cpu_values)
        
        if avg_cpu > 90:
            recommendations.append("⚠️ CPU使用过高，考虑优化算法或增加CPU")
        
        return recommendations