# 添加监控脚本 monitor.py
import psutil
import time

def monitor_resources():
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"CPU使用率: {cpu_percent}%")
        print(f"内存使用: {memory.percent}% ({memory.used/1024**3:.1f}GB)")
        print(f"磁盘使用: {disk.percent}%")
        print("-" * 40)
        
        time.sleep(60)