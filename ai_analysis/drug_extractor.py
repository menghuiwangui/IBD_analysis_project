# src/ai_analysis/drug_extractor.py
import re
import pandas as pd
from collections import Counter
import json
from config.settings import RESULTS_DIR

class DrugGeneExtractor:
    def __init__(self):
        # IBD相关药物
        self.drugs = {
            '5-ASA': ['mesalamine', 'mesalazine', 'sulfasalazine', 'olsalazine', 'balsalazide'],
            'Corticosteroids': ['prednisone', 'budesonide', 'methylprednisolone', 'hydrocortisone'],
            'Immunomodulators': ['azathioprine', 'mercaptopurine', 'methotrexate', 'cyclosporine', 'tacrolimus'],
            'Anti-TNF': ['infliximab', 'adalimumab', 'golimumab', 'certolizumab'],
            'Anti-integrin': ['vedolizumab', 'natalizumab'],
            'Anti-IL-12/23': ['ustekinumab', 'risankizumab'],
            'JAK inhibitors': ['tofacitinib', 'upadacitinib', 'filgotinib'],
            'Antibiotics': ['metronidazole', 'ciprofloxacin', 'rifaximin']
        }
        
        # 扁平化药物列表
        self.drug_list = []
        for category, drugs in self.drugs.items():
            self.drug_list.extend(drugs)
        
        # IBD相关基因
        self.genes = [
            'NOD2', 'IL23R', 'ATG16L1', 'CARD9', 'TNFAIP3',
            'IRGM', 'PTPN2', 'IL10', 'IL10RA', 'IL10RB',
            'XBP1', 'JAK2', 'STAT3', 'STAT4', 'CCR6',
            'FUT2', 'LACC1', 'GPR35', 'TNF', 'IL6',
            'IL12B', 'IL17A', 'IL22', 'IL1B', 'TLR4'
        ]
        
        # 通路
        self.pathways = [
            'TNF-alpha', 'IL-12/IL-23', 'JAK-STAT', 'NF-kappaB',
            'autophagy', 'gut barrier', 'mucosal immunity',
            'cytokine signaling', 'T cell', 'B cell', 'macrophage',
            'inflammasome', 'oxidative stress'
        ]
    
    def extract_entities(self, text):
        """提取实体"""
        if not isinstance(text, str):
            return {'drugs': [], 'genes': [], 'pathways': []}
        
        text_lower = text.lower()
        
        # 提取药物
        found_drugs = []
        for drug in self.drug_list:
            if re.search(rf'\b{re.escape(drug.lower())}\b', text_lower):
                found_drugs.append(drug)
        
        # 提取基因
        found_genes = []
        for gene in self.genes:
            if re.search(rf'\b{re.escape(gene)}\b', text, re.IGNORECASE):
                found_genes.append(gene)
        
        # 提取通路
        found_pathways = []
        for pathway in self.pathways:
            pathway_lower = pathway.lower()
            if pathway_lower in text_lower:
                found_pathways.append(pathway)
        
        return {
            'drugs': list(set(found_drugs)),
            'genes': list(set(found_genes)),
            'pathways': list(set(found_pathways))
        }
    
    def extract_from_df(self, df, text_column='text'):
        """从DataFrame提取"""
        print(f"🔍 Extracting entities from {len(df)} articles...")
        
        results = []
        for idx, row in df.iterrows():
            if idx % 100 == 0 and idx > 0:
                print(f"   Processed {idx}/{len(df)} articles...")
            
            text = row[text_column] if text_column in row else ''
            entities = self.extract_entities(text)
            results.append(entities)
        
        return results
    
    def analyze(self, df, text_column='text'):
        """分析并添加到DataFrame"""
        # 提取实体
        results = self.extract_from_df(df, text_column)
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 合并到原始数据
        df_with_entities = df.copy()
        df_with_entities['drugs'] = results_df['drugs']
        df_with_entities['genes'] = results_df['genes']
        df_with_entities['pathways'] = results_df['pathways']
        
        # 计算出现次数
        df_with_entities['drug_count'] = df_with_entities['drugs'].apply(len)
        df_with_entities['gene_count'] = df_with_entities['genes'].apply(len)
        df_with_entities['pathway_count'] = df_with_entities['pathways'].apply(len)
        
        return df_with_entities
    
    def get_statistics(self, df):
        """获取统计信息"""
        all_drugs = []
        all_genes = []
        all_pathways = []
        
        for drugs in df['drugs']:
            all_drugs.extend(drugs)
        for genes in df['genes']:
            all_genes.extend(genes)
        for pathways in df['pathways']:
            all_pathways.extend(pathways)
        
        drug_counts = Counter(all_drugs)
        gene_counts = Counter(all_genes)
        pathway_counts = Counter(all_pathways)
        
        # 按药物类别统计
        drug_by_category = {}
        for category, drugs in self.drugs.items():
            count = 0
            for drug in drugs:
                if drug in drug_counts:
                    count += drug_counts[drug]
            if count > 0:
                drug_by_category[category] = count
        
        return {
            'drugs': drug_counts.most_common(20),
            'drugs_by_category': sorted(drug_by_category.items(), key=lambda x: x[1], reverse=True),
            'genes': gene_counts.most_common(20),
            'pathways': pathway_counts.most_common(15)
        }
    
    def save_results(self, df, stats):
        """保存结果"""
        # 保存DataFrame
        output_path = RESULTS_DIR / 'ibd_with_entities.csv'
        df.to_csv(output_path, index=False)
        print(f"✅ Entities saved to {output_path}")
        
        # 保存统计
        stats_path = RESULTS_DIR / 'entity_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 保存文本报告
        report_path = RESULTS_DIR / 'entity_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("IBD LITERATURE ENTITY ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total articles analyzed: {len(df)}\n\n")
            
            f.write("TOP DRUGS BY CATEGORY:\n")
            f.write("-" * 30 + "\n")
            for category, count in stats['drugs_by_category']:
                f.write(f"{category}: {count}\n")
            
            f.write("\nTOP DRUGS:\n")
            f.write("-" * 30 + "\n")
            for drug, count in stats['drugs'][:10]:
                f.write(f"{drug}: {count}\n")
            
            f.write("\nTOP GENES:\n")
            f.write("-" * 30 + "\n")
            for gene, count in stats['genes'][:10]:
                f.write(f"{gene}: {count}\n")
            
            f.write("\nTOP PATHWAYS:\n")
            f.write("-" * 30 + "\n")
            for pathway, count in stats['pathways'][:10]:
                f.write(f"{pathway}: {count}\n")
        
        print(f"✅ Report saved to {report_path}")
        return output_path