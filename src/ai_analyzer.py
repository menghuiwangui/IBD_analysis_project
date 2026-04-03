import paddle
import paddlenlp as ppnlp
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import os

class AIAnalyzer:
    def __init__(self, model_name="ernie-3.0-base-zh"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=4)
    
    def preprocess_text(self, text, max_length=512):
        """预处理文本"""
        return self.tokenizer(text, max_length=max_length, padding='max_length', truncation=True)
    
    def analyze_treatment(self, abstracts):
        """分析治疗方法"""
        # 这里使用简单的规则匹配，实际应用中可以使用更复杂的模型
        treatment_keywords = ["治疗", "疗法", "手术", "药物", "治疗方案"]
        treatments = []
        
        for abstract in abstracts:
            if not abstract:
                treatments.append([])
                continue
            
            found = []
            for keyword in treatment_keywords:
                if keyword in abstract:
                    # 提取包含关键词的句子
                    sentences = abstract.split('。')
                    for sentence in sentences:
                        if keyword in sentence:
                            found.append(sentence.strip())
            treatments.append(found)
        
        return treatments
    
    def analyze_drugs(self, abstracts):
        """分析药物"""
        # 常见IBD相关药物
        drug_keywords = ["美沙拉嗪", "泼尼松", "英夫利昔单抗", "阿达木单抗", "硫唑嘌呤", "甲氨蝶呤"]
        drugs = []
        
        for abstract in abstracts:
            if not abstract:
                drugs.append([])
                continue
            
            found = []
            for drug in drug_keywords:
                if drug in abstract:
                    found.append(drug)
            drugs.append(found)
        
        return drugs
    
    def analyze_genes(self, abstracts):
        """分析基因"""
        # 常见IBD相关基因
        gene_keywords = ["NOD2", "IL23R", "ATG16L1", "IRGM", "FUT2"]
        genes = []
        
        for abstract in abstracts:
            if not abstract:
                genes.append([])
                continue
            
            found = []
            for gene in gene_keywords:
                if gene in abstract:
                    found.append(gene)
            genes.append(found)
        
        return genes
    
    def analyze_pathways(self, abstracts):
        """分析通路"""
        pathway_keywords = ["炎症通路", "免疫通路", "信号通路", "细胞因子", "NF-κB", "MAPK"]
        pathways = []
        
        for abstract in abstracts:
            if not abstract:
                pathways.append([])
                continue
            
            found = []
            for pathway in pathway_keywords:
                if pathway in abstract:
                    found.append(pathway)
            pathways.append(found)
        
        return pathways
    
    def analyze_papers(self, df):
        """分析所有文献"""
        abstracts = df['abstract'].tolist()
        
        print("正在分析治疗方法...")
        treatments = self.analyze_treatment(abstracts)
        
        print("正在分析药物...")
        drugs = self.analyze_drugs(abstracts)
        
        print("正在分析基因...")
        genes = self.analyze_genes(abstracts)
        
        print("正在分析通路...")
        pathways = self.analyze_pathways(abstracts)
        
        # 添加分析结果到DataFrame
        df['analyzed_treatments'] = treatments
        df['analyzed_drugs'] = drugs
        df['analyzed_genes'] = genes
        df['analyzed_pathways'] = pathways
        
        return df
