import requests
import json
import time
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class PebmebAPI:
    def __init__(self, email=None):
        self.email = email or os.getenv('PEBMEB_EMAIL')
        self.base_url = "https://api.pebmeb.com/v1"
        self.headers = {
            "Authorization": f"Email {self.email}",
            "Content-Type": "application/json"
        }
    
    def search_ibd_papers(self, query="IBD", limit=100, offset=0):
        """搜索IBD相关文献"""
        url = f"{self.base_url}/papers/search"
        payload = {
            "query": query,
            "limit": limit,
            "offset": offset
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API调用失败: {e}")
            return None
    
    def get_all_ibd_papers(self, query="IBD", batch_size=100):
        """获取所有IBD相关文献"""
        all_papers = []
        offset = 0
        
        while True:
            print(f"正在获取第 {offset} 到 {offset + batch_size - 1} 条文献...")
            result = self.search_ibd_papers(query, batch_size, offset)
            
            if not result or not result.get('papers'):
                break
            
            papers = result['papers']
            all_papers.extend(papers)
            
            if len(papers) < batch_size:
                break
            
            offset += batch_size
            time.sleep(1)  # 避免API请求过于频繁
        
        return all_papers
