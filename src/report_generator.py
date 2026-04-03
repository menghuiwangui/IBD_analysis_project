import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_statistics(self, df):
        """生成统计信息"""
        stats = {
            "total_papers": len(df),
            "papers_with_treatment": len(df[df['analyzed_treatments'].apply(len) > 0]),
            "papers_with_drugs": len(df[df['analyzed_drugs'].apply(len) > 0]),
            "papers_with_genes": len(df[df['analyzed_genes'].apply(len) > 0]),
            "papers_with_pathways": len(df[df['analyzed_pathways'].apply(len) > 0])
        }
        return stats
    
    def analyze_drug_frequency(self, df):
        """分析药物频率"""
        drug_counts = {}
        for drugs in df['analyzed_drugs']:
            for drug in drugs:
                if drug in drug_counts:
                    drug_counts[drug] += 1
                else:
                    drug_counts[drug] = 1
        return sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
    
    def analyze_gene_frequency(self, df):
        """分析基因频率"""
        gene_counts = {}
        for genes in df['analyzed_genes']:
            for gene in genes:
                if gene in gene_counts:
                    gene_counts[gene] += 1
                else:
                    gene_counts[gene] = 1
        return sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)
    
    def analyze_pathway_frequency(self, df):
        """分析通路频率"""
        pathway_counts = {}
        for pathways in df['analyzed_pathways']:
            for pathway in pathways:
                if pathway in pathway_counts:
                    pathway_counts[pathway] += 1
                else:
                    pathway_counts[pathway] = 1
        return sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True)
    
    def plot_drug_frequency(self, drug_freq):
        """绘制药物频率图"""
        if not drug_freq:
            return
        
        drugs, counts = zip(*drug_freq[:10])  # 只显示前10个
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(drugs), y=list(counts))
        plt.title('Top 10 药物频率')
        plt.xlabel('药物')
        plt.ylabel('频率')
        plt.xticks(rotation=45)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'drug_frequency.png')
        plt.savefig(filepath)
        plt.close()
        return filepath
    
    def plot_gene_frequency(self, gene_freq):
        """绘制基因频率图"""
        if not gene_freq:
            return
        
        genes, counts = zip(*gene_freq[:10])  # 只显示前10个
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(genes), y=list(counts))
        plt.title('Top 10 基因频率')
        plt.xlabel('基因')
        plt.ylabel('频率')
        plt.xticks(rotation=45)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'gene_frequency.png')
        plt.savefig(filepath)
        plt.close()
        return filepath
    
    def plot_pathway_frequency(self, pathway_freq):
        """绘制通路频率图"""
        if not pathway_freq:
            return
        
        pathways, counts = zip(*pathway_freq[:10])  # 只显示前10个
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(pathways), y=list(counts))
        plt.title('Top 10 通路频率')
        plt.xlabel('通路')
        plt.ylabel('频率')
        plt.xticks(rotation=45)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'pathway_frequency.png')
        plt.savefig(filepath)
        plt.close()
        return filepath
    
    def generate_report(self, df):
        """生成完整报告"""
        stats = self.generate_statistics(df)
        drug_freq = self.analyze_drug_frequency(df)
        gene_freq = self.analyze_gene_frequency(df)
        pathway_freq = self.analyze_pathway_frequency(df)
        
        # 生成可视化图表
        self.plot_drug_frequency(drug_freq)
        self.plot_gene_frequency(gene_freq)
        self.plot_pathway_frequency(pathway_freq)
        
        # 生成文本报告
        report = f"""# IBD 前沿研究分析报告

## 统计信息
- 总文献数: {stats['total_papers']}
- 包含治疗方法的文献数: {stats['papers_with_treatment']}
- 包含药物的文献数: {stats['papers_with_drugs']}
- 包含基因的文献数: {stats['papers_with_genes']}
- 包含通路的文献数: {stats['papers_with_pathways']}

## 药物分析
### 高频药物:
"""
        
        for drug, count in drug_freq[:10]:
            report += f"- {drug}: {count} 篇文献\n"
        
        report += f"""

## 基因分析
### 高频基因:
"""
        
        for gene, count in gene_freq[:10]:
            report += f"- {gene}: {count} 篇文献\n"
        
        report += f"""

## 通路分析
### 高频通路:
"""
        
        for pathway, count in pathway_freq[:10]:
            report += f"- {pathway}: {count} 篇文献\n"
        
        # 保存报告
        filepath = os.path.join(self.output_dir, 'ibd_analysis_report.md')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"报告已生成: {filepath}")
        return filepath
