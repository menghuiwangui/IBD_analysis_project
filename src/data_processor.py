import pandas as pd
import json
import os

class DataProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save_papers_to_json(self, papers, filename="ibd_papers.json"):
        """将文献数据保存为JSON文件"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        print(f"文献数据已保存到: {filepath}")
        return filepath
    
    def load_papers_from_json(self, filename="ibd_papers.json"):
        """从JSON文件加载文献数据"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        print(f"成功加载 {len(papers)} 篇文献")
        return papers
    
    def papers_to_dataframe(self, papers):
        """将文献数据转换为DataFrame"""
        if not papers:
            return pd.DataFrame()
        
        # 提取需要的字段
        data = []
        for paper in papers:
            item = {
                "id": paper.get("id"),
                "title": paper.get("title"),
                "authors": ", ".join(paper.get("authors", [])),
                "abstract": paper.get("abstract"),
                "publication_date": paper.get("publication_date"),
                "journal": paper.get("journal"),
                "keywords": ", ".join(paper.get("keywords", [])),
                "url": paper.get("url"),
                "treatment": paper.get("treatment", ""),
                "drugs": ", ".join(paper.get("drugs", [])),
                "genes": ", ".join(paper.get("genes", [])),
                "pathways": ", ".join(paper.get("pathways", []))
            }
            data.append(item)
        
        df = pd.DataFrame(data)
        return df
    
    def save_dataframe_to_csv(self, df, filename="ibd_papers.csv"):
        """将DataFrame保存为CSV文件"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"数据已保存到: {filepath}")
        return filepath
