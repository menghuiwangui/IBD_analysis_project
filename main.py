from src.api import PebmebAPI
from src.data_processor import DataProcessor
from src.ai_analyzer import AIAnalyzer
from src.report_generator import ReportGenerator
import os

class IBDResearchAnalyzer:
    def __init__(self):
        self.api = PebmebAPI()
        self.processor = DataProcessor()
        self.analyzer = AIAnalyzer()
        self.report_generator = ReportGenerator()
    
    def run(self):
        """运行完整的分析流程"""
        # 1. 获取IBD相关文献
        print("正在获取IBD相关文献...")
        papers = self.api.get_all_ibd_papers()
        
        if not papers:
            print("未获取到文献数据，请检查API配置")
            return
        
        print(f"成功获取 {len(papers)} 篇IBD相关文献")
        
        # 2. 保存和处理数据
        print("正在处理数据...")
        self.processor.save_papers_to_json(papers)
        df = self.processor.papers_to_dataframe(papers)
        self.processor.save_dataframe_to_csv(df)
        
        # 3. AI分析
        print("正在进行AI分析...")
        analyzed_df = self.analyzer.analyze_papers(df)
        
        # 4. 生成报告
        print("正在生成分析报告...")
        self.report_generator.generate_report(analyzed_df)
        
        print("分析完成！")

if __name__ == "__main__":
    analyzer = IBDResearchAnalyzer()
    analyzer.run()
