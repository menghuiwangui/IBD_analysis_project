# src/ai_analysis/entity_extractor_enhanced.py
"""
增强版实体提取器，识别药物、基因、通路等
"""
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set, Tuple
import json
from collections import Counter, defaultdict
import logging
from pathlib import Path
import spacy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 尝试加载spaCy模型
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    print("✅ spaCy 可用，将用于高级实体识别")
except:
    SPACY_AVAILABLE = False
    print("⚠️ spaCy 不可用，将使用基于规则的实体识别")

from config.settings_paddle import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedEntityExtractor:
    """增强版实体提取器"""
    
    def __init__(self, use_spacy: bool = True):
        """
        初始化实体提取器
        
        Args:
            use_spacy: 是否使用spaCy进行实体识别
        """
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        # 预定义实体词典
        self._load_entity_dictionaries()
        
        # 正则表达式模式
        self._compile_patterns()
        
        logger.info(f"初始化实体提取器 (使用spaCy: {self.use_spacy})")
    
    def _load_entity_dictionaries(self):
        """加载实体词典"""
        
        # IBD相关药物词典（分类）
        self.drug_categories = {
            '5-ASA': [
                'mesalamine', 'mesalazine', 'sulfasalazine', 'olsalazine', 'balsalazide',
                'pentasa', 'asacol', 'lialda', 'apriso', 'delzicol', 'salofalk'
            ],
            'Corticosteroids': [
                'prednisone', 'prednisolone', 'budesonide', 'methylprednisolone',
                'hydrocortisone', 'cortisone', 'dexamethasone', 'betamethasone'
            ],
            'Immunomodulators': [
                'azathioprine', 'mercaptopurine', 'methotrexate', 'cyclosporine',
                'tacrolimus', 'mycophenolate', 'mycophenolic acid', 'tacrolimus'
            ],
            'Anti-TNF': [
                'infliximab', 'adalimumab', 'golimumab', 'certolizumab',
                'remicade', 'humira', 'simponi', 'cimzia'
            ],
            'Anti-integrin': [
                'vedolizumab', 'natalizumab', 'entyvio', 'tysabri'
            ],
            'Anti-IL-12/23': [
                'ustekinumab', 'risankizumab', 'stelara', 'skyrizi'
            ],
            'JAK inhibitors': [
                'tofacitinib', 'upadacitinib', 'filgotinib', 'peficitinib',
                'xeljanz', 'rinvoq', 'jyseleca'
            ],
            'Antibiotics': [
                'metronidazole', 'ciprofloxacin', 'rifaximin', 'vancomycin',
                'amoxicillin', 'doxycycline', 'clarithromycin'
            ],
            'Probiotics': [
                'vsl#3', 'e. coli nissle', 'saccharomyces boulardii',
                'lactobacillus', 'bifidobacterium'
            ]
        }
        
        # 扁平化药物列表
        self.drug_list = []
        for category, drugs in self.drug_categories.items():
            self.drug_list.extend(drugs)
        
        # 基因词典（IBD相关基因）
        self.genes = [
            # NOD2通路
            'NOD2', 'CARD15', 'RIPK2', 'XIAP',
            
            # 自噬相关
            'ATG16L1', 'IRGM', 'ULK1', 'LC3',
            
            # IL-23/Th17通路
            'IL23R', 'IL12B', 'IL23A', 'STAT3', 'STAT4',
            'JAK2', 'TYK2', 'RORC', 'IL17A', 'IL17F',
            
            # TNF通路
            'TNF', 'TNFAIP3', 'TNFRSF1A', 'TNFRSF1B',
            
            # 肠道屏障
            'HNF4A', 'CDH1', 'OCLN', 'CLDN2',
            
            # 其他重要基因
            'PTPN2', 'PTPN22', 'IL10', 'IL10RA', 'IL10RB',
            'FUT2', 'LACC1', 'GPR35', 'CCR6', 'CCR9',
            'ITLN1', 'CARD9', 'ZNF365', 'ECM1',
            
            # 免疫调节
            'IFNG', 'IL6', 'IL1B', 'IL22', 'TLR4', 'TLR9',
            
            # 最近发现的基因
            'ADCY7', 'DOK3', 'PRDM1', 'PLA2G2E', 'C1orf106'
        ]
        
        # 通路词典
        self.pathways = {
            'Innate Immunity': [
                'NOD-like receptor signaling', 'TLR signaling', 'NF-kappaB signaling',
                'Inflammasome activation', 'Autophagy'
            ],
            'Adaptive Immunity': [
                'T cell differentiation', 'Th17 pathway', 'Treg development',
                'B cell activation', 'Cytokine signaling'
            ],
            'Barrier Function': [
                'Gut barrier integrity', 'Mucosal defense', 'Epithelial repair',
                'Tight junction regulation'
            ],
            'Cytokine Signaling': [
                'TNF-alpha signaling', 'IL-23/IL-17 axis', 'IL-10 signaling',
                'IFN-gamma signaling', 'IL-6 signaling'
            ],
            'Metabolic Pathways': [
                'JAK-STAT signaling', 'PI3K-AKT signaling', 'MAPK signaling',
                'Wnt signaling', 'Notch signaling'
            ]
        }
        
        # 疾病术语
        self.diseases = [
            'Crohn\'s disease', 'ulcerative colitis', 'inflammatory bowel disease',
            'IBD-unclassified', 'indeterminate colitis', 'microscopic colitis',
            'collagenous colitis', 'lymphocytic colitis', 'pouchitis'
        ]
        
        # 细胞类型
        self.cell_types = [
            'T cell', 'B cell', 'macrophage', 'dendritic cell', 'neutrophil',
            'eosinophil', 'epithelial cell', 'goblet cell', 'paneth cell',
            'enterocyte', 'fibroblast', 'myofibroblast'
        ]
        
        # 生物标志物
        self.biomarkers = [
            'CRP', 'ESR', 'calprotectin', 'lactoferrin', 'S100A12',
            'MMP9', 'TNF-alpha', 'IL-6', 'IL-8', 'IL-17', 'IL-22',
            'fecal calprotectin', 'serum CRP'
        ]
        
        # 治疗靶点
        self.targets = [
            'TNF-alpha', 'IL-12/23', 'alpha4beta7 integrin', 'JAK',
            'S1P receptor', 'IL-6 receptor', 'IL-1beta'
        ]
        
        logger.info(f"加载实体词典完成: "
                   f"{len(self.drug_list)} 药物, "
                   f"{len(self.genes)} 基因, "
                   f"{sum(len(v) for v in self.pathways.values())} 通路")
    
    def _compile_patterns(self):
        """编译正则表达式模式"""
        # 药物模式（支持不同拼写）
        self.drug_patterns = {}
        for category, drugs in self.drug_categories.items():
            patterns = []
            for drug in drugs:
                # 创建不区分大小写的模式，支持复数形式
                pattern = rf'\b{re.escape(drug)}\w*\b'
                patterns.append(pattern)
            self.drug_patterns[category] = re.compile('|'.join(patterns), re.IGNORECASE)
        
        # 基因模式（大写，可能带数字和连字符）
        gene_patterns = [rf'\b{re.escape(gene)}\b' for gene in self.genes]
        self.gene_pattern = re.compile('|'.join(gene_patterns))
        
        # 通路模式
        self.pathway_patterns = {}
        for category, pathways in self.pathways.items():
            patterns = []
            for pathway in pathways:
                # 支持多种表达方式
                pattern = rf'\b{re.escape(pathway.lower())}\b'
                patterns.append(pattern)
            self.pathway_patterns[category] = re.compile('|'.join(patterns), re.IGNORECASE)
    
    def extract_entities_spacy(self, text: str) -> Dict[str, List[str]]:
        """使用spaCy提取实体"""
        if not self.use_spacy or not text:
            return self.extract_entities_regex(text)
        
        doc = nlp(text)
        
        entities = {
            'drugs': set(),
            'genes': set(),
            'pathways': set(),
            'diseases': set(),
            'cell_types': set(),
            'biomarkers': set(),
            'targets': set()
        }
        
        # 提取命名实体
        for ent in doc.ents:
            ent_text = ent.text.lower()
            
            # 检查是否为药物
            if ent.label_ in ['CHEM', 'DRUG']:
                for category, pattern in self.drug_patterns.items():
                    if pattern.search(ent_text):
                        entities['drugs'].add(ent.text)
                        break
            
            # 检查是否为疾病
            elif ent.label_ == 'DISEASE':
                for disease in self.diseases:
                    if disease.lower() in ent_text:
                        entities['diseases'].add(disease)
            
            # 检查是否为基因/蛋白质
            elif ent.label_ in ['GENE', 'PROTEIN']:
                # 检查是否在基因列表中
                if self.gene_pattern.search(ent.text):
                    entities['genes'].add(ent.text)
        
        # 使用规则提取其他实体
        entities.update(self._extract_with_regex(text))
        
        # 转换为列表
        return {k: list(v) for k, v in entities.items()}
    
    def extract_entities_regex(self, text: str) -> Dict[str, List[str]]:
        """使用正则表达式提取实体"""
        if not text:
            return {k: [] for k in ['drugs', 'genes', 'pathways', 'diseases', 
                                   'cell_types', 'biomarkers', 'targets']}
        
        text_lower = text.lower()
        
        entities = {
            'drugs': set(),
            'genes': set(),
            'pathways': set(),
            'diseases': set(),
            'cell_types': set(),
            'biomarkers': set(),
            'targets': set()
        }
        
        # 提取药物
        for category, pattern in self.drug_patterns.items():
            matches = pattern.findall(text_lower)
            if matches:
                # 去重并规范化
                for match in matches:
                    # 找到匹配的原始药物名
                    for drug in self.drug_categories[category]:
                        if drug.lower() in match.lower():
                            entities['drugs'].add(drug)
                            break
        
        # 提取基因（保持原始大小写）
        for match in re.finditer(r'\b[A-Z][A-Z0-9\-]+\b', text):
            gene_candidate = match.group()
            # 检查是否在基因列表中
            if gene_candidate in self.genes:
                entities['genes'].add(gene_candidate)
        
        # 提取通路
        for category, pattern in self.pathway_patterns.items():
            matches = pattern.findall(text_lower)
            for match in matches:
                # 找到匹配的原始通路名
                for pathway in self.pathways[category]:
                    if pathway.lower() in match.lower():
                        entities['pathways'].add(pathway)
                        break
        
        # 提取疾病
        for disease in self.diseases:
            if disease.lower() in text_lower:
                entities['diseases'].add(disease)
        
        # 提取细胞类型
        for cell_type in self.cell_types:
            if cell_type.lower() in text_lower:
                entities['cell_types'].add(cell_type)
        
        # 提取生物标志物
        for biomarker in self.biomarkers:
            if biomarker.lower() in text_lower:
                entities['biomarkers'].add(biomarker)
        
        # 提取治疗靶点
        for target in self.targets:
            if target.lower() in text_lower:
                entities['targets'].add(target)
        
        # 转换为列表
        return {k: list(v) for k, v in entities.items()}
    
    def _extract_with_regex(self, text: str) -> Dict[str, Set[str]]:
        """使用正则表达式辅助提取"""
        text_lower = text.lower()
        
        entities = {
            'drugs': set(),
            'genes': set(),
            'pathways': set(),
            'diseases': set(),
            'cell_types': set(),
            'biomarkers': set(),
            'targets': set()
        }
        
        # 这里可以添加更多正则表达式规则
        return entities
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取实体（自动选择方法）"""
        if self.use_spacy:
            return self.extract_entities_spacy(text)
        else:
            return self.extract_entities_regex(text)
    
    def extract_entities_batch(self, texts: List[str], batch_size: int = 1000) -> List[Dict[str, List[str]]]:
        """批量提取实体"""
        logger.info(f"批量提取实体，文本数量: {len(texts)}")
        
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="提取实体"):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                entities = self.extract_entities(text)
                batch_results.append(entities)
            
            results.extend(batch_results)
            
            # 每10个批次打印进度
            if (i // batch_size) % 10 == 0:
                logger.info(f"  已处理 {min(i + batch_size, len(texts))}/{len(texts)} 篇文章")
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'full_text') -> pd.DataFrame:
        """分析DataFrame中的所有文本"""
        logger.info(f"开始分析DataFrame，总行数: {len(df)}")
        
        # 提取文本
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # 批量提取实体
        entities_list = self.extract_entities_batch(texts)
        
        # 转换为DataFrame
        entities_df = pd.DataFrame(entities_list)
        
        # 合并到原始DataFrame
        df_with_entities = df.copy()
        for col in entities_df.columns:
            df_with_entities[col] = entities_df[col]
        
        # 计算实体数量
        entity_count_cols = ['drugs', 'genes', 'pathways', 'diseases', 
                           'cell_types', 'biomarkers', 'targets']
        for col in entity_count_cols:
            if col in df_with_entities.columns:
                df_with_entities[f'{col}_count'] = df_with_entities[col].apply(len)
        
        logger.info("✅ 实体提取完成")
        return df_with_entities
    
    def get_entity_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取实体统计信息"""
        logger.info("计算实体统计信息...")
        
        statistics = {}
        
        # 计算每个实体的频率
        entity_columns = ['drugs', 'genes', 'pathways', 'diseases', 
                         'cell_types', 'biomarkers', 'targets']
        
        for col in entity_columns:
            if col in df.columns:
                # 展开所有实体
                all_entities = []
                for entities in df[col]:
                    if isinstance(entities, list):
                        all_entities.extend(entities)
                
                # 计算频率
                counter = Counter(all_entities)
                
                # 按类别分组（对药物）
                if col == 'drugs':
                    by_category = defaultdict(int)
                    for entity, count in counter.items():
                        for category, drugs in self.drug_categories.items():
                            if entity in drugs:
                                by_category[category] += count
                                break
                    statistics[f'{col}_by_category'] = dict(sorted(
                        by_category.items(), key=lambda x: x[1], reverse=True
                    ))
                
                # 通用统计
                statistics[col] = {
                    'total_unique': len(counter),
                    'total_mentions': sum(counter.values()),
                    'top_20': counter.most_common(20),
                    'all_entities': dict(counter.most_common(100))  # 只保存前100个
                }
        
        # 计算相关性
        statistics['correlations'] = self._calculate_correlations(df)
        
        # 计算时间趋势（如果存在年份列）
        if 'year' in df.columns:
            statistics['temporal_trends'] = self._calculate_temporal_trends(df)
        
        logger.info("✅ 统计信息计算完成")
        return statistics
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算实体之间的相关性"""
        correlations = {}
        
        try:
            # 药物-基因共现
            if 'drugs' in df.columns and 'genes' in df.columns:
                drug_gene_cooccur = defaultdict(Counter)
                
                for _, row in df.iterrows():
                    drugs = row['drugs'] if isinstance(row['drugs'], list) else []
                    genes = row['genes'] if isinstance(row['genes'], list) else []
                    
                    for drug in drugs:
                        for gene in genes:
                            drug_gene_cooccur[drug][gene] += 1
                
                # 找出最强的关联
                strong_associations = []
                for drug, gene_counts in drug_gene_cooccur.items():
                    for gene, count in gene_counts.most_common(5):
                        if count >= 5:  # 至少共同出现5次
                            strong_associations.append({
                                'drug': drug,
                                'gene': gene,
                                'cooccurrence': count
                            })
                
                correlations['drug_gene'] = sorted(
                    strong_associations, 
                    key=lambda x: x['cooccurrence'], 
                    reverse=True
                )[:50]  # 只保存前50个
            
            # 通路-疾病关联
            if 'pathways' in df.columns and 'diseases' in df.columns:
                pathway_disease_cooccur = defaultdict(Counter)
                
                for _, row in df.iterrows():
                    pathways = row['pathways'] if isinstance(row['pathways'], list) else []
                    diseases = row['diseases'] if isinstance(row['diseases'], list) else []
                    
                    for pathway in pathways:
                        for disease in diseases:
                            pathway_disease_cooccur[pathway][disease] += 1
                
                correlations['pathway_disease'] = [
                    {
                        'pathway': pathway,
                        'disease': disease,
                        'cooccurrence': count
                    }
                    for pathway, disease_counts in pathway_disease_cooccur.items()
                    for disease, count in disease_counts.most_common(3)
                    if count >= 3
                ]
        
        except Exception as e:
            logger.warning(f"计算相关性时出错: {e}")
        
        return correlations
    
    def _calculate_temporal_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算时间趋势"""
        trends = {}
        
        try:
            # 按年份分组
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df[df['year'].notna()]
            
            entity_columns = ['drugs', 'genes', 'pathways']
            
            for col in entity_columns:
                if col in df.columns:
                    yearly_trends = {}
                    
                    for year in sorted(df['year'].unique()):
                        year_df = df[df['year'] == year]
                        
                        # 展开实体
                        all_entities = []
                        for entities in year_df[col]:
                            if isinstance(entities, list):
                                all_entities.extend(entities)
                        
                        counter = Counter(all_entities)
                        yearly_trends[int(year)] = dict(counter.most_common(10))
                    
                    trends[col] = yearly_trends
        
        except Exception as e:
            logger.warning(f"计算时间趋势时出错: {e}")
        
        return trends
    
    def save_analysis_results(self, df: pd.DataFrame, stats: Dict[str, Any], 
                             output_prefix: str = "ibd_entities") -> Dict[str, Path]:
        """保存分析结果"""
        logger.info("保存分析结果...")
        
        output_files = {}
        
        # 1. 保存带有实体的DataFrame
        parquet_path = RESULTS_DIR / f"{output_prefix}_detailed.parquet"
        df.to_parquet(parquet_path, index=False)
        output_files['data'] = parquet_path
        logger.info(f"✅ 数据已保存: {parquet_path}")
        
        # 2. 保存统计信息
        stats_path = RESULTS_DIR / f"{output_prefix}_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        output_files['statistics'] = stats_path
        logger.info(f"✅ 统计信息已保存: {stats_path}")
        
        # 3. 保存文本报告
        report_path = RESULTS_DIR / f"{output_prefix}_report.txt"
        self._generate_text_report(stats, report_path)
        output_files['report'] = report_path
        logger.info(f"✅ 文本报告已保存: {report_path}")
        
        # 4. 保存CSV摘要
        summary_path = RESULTS_DIR / f"{output_prefix}_summary.csv"
        self._generate_summary_csv(df, stats, summary_path)
        output_files['summary'] = summary_path
        
        # 5. 保存可视化
        try:
            self._create_visualizations(df, stats, output_prefix)
            output_files['visualizations'] = RESULTS_DIR / f"{output_prefix}_visualizations"
        except Exception as e:
            logger.warning(f"创建可视化时出错: {e}")
        
        return output_files
    
    def _generate_text_report(self, stats: Dict[str, Any], report_path: Path):
        """生成文本报告"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("IBD文献实体分析报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📊 总体统计\n")
            f.write("-" * 40 + "\n")
            
            for entity_type in ['drugs', 'genes', 'pathways']:
                if entity_type in stats:
                    entity_stats = stats[entity_type]
                    f.write(f"\n{entity_type.upper()}:\n")
                    f.write(f"  唯一实体数: {entity_stats['total_unique']}\n")
                    f.write(f"  总提及次数: {entity_stats['total_mentions']}\n")
                    
                    f.write(f"  前10个最常提及的{entity_type}:\n")
                    for i, (entity, count) in enumerate(entity_stats['top_20'][:10], 1):
                        f.write(f"    {i:2d}. {entity:<30} {count:>6} 次\n")
            
            # 药物分类统计
            if 'drugs_by_category' in stats:
                f.write("\n💊 药物分类统计:\n")
                for category, count in stats['drugs_by_category'].items():
                    f.write(f"  {category:<20} {count:>6} 次提及\n")
            
            # 相关性分析
            if 'correlations' in stats and 'drug_gene' in stats['correlations']:
                f.write("\n🔗 药物-基因关联分析:\n")
                for i, assoc in enumerate(stats['correlations']['drug_gene'][:10], 1):
                    f.write(f"  {i:2d}. {assoc['drug']:<20} - {assoc['gene']:<15} "
                           f"(共同出现: {assoc['cooccurrence']} 次)\n")
            
            # 时间趋势
            if 'temporal_trends' in stats and 'drugs' in stats['temporal_trends']:
                f.write("\n📈 药物研究时间趋势:\n")
                yearly_trends = stats['temporal_trends']['drugs']
                for year in sorted(yearly_trends.keys())[-5:]:  # 最近5年
                    f.write(f"  {year}: {', '.join(yearly_trends[year].keys())}\n")
    
    def _generate_summary_csv(self, df: pd.DataFrame, stats: Dict[str, Any], summary_path: Path):
        """生成CSV摘要"""
        summary_data = []
        
        # 实体频率表
        for entity_type in ['drugs', 'genes', 'pathways']:
            if entity_type in stats:
                for entity, count in stats[entity_type]['top_20']:
                    summary_data.append({
                        'entity_type': entity_type,
                        'entity_name': entity,
                        'mention_count': count,
                        'percentage': count / stats[entity_type]['total_mentions'] * 100
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)
    
    def _create_visualizations(self, df: pd.DataFrame, stats: Dict[str, Any], output_prefix: str):
        """创建可视化图表"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置样式
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. 药物类别分布
            if 'drugs_by_category' in stats:
                plt.figure(figsize=(10, 6))
                categories = list(stats['drugs_by_category'].keys())
                counts = list(stats['drugs_by_category'].values())
                
                bars = plt.barh(categories, counts)
                plt.xlabel('提及次数')
                plt.title('IBD药物类别分布')
                plt.tight_layout()
                plt.savefig(RESULTS_DIR / f"{output_prefix}_drug_categories.png", dpi=300)
                plt.close()
            
            # 2. 实体频率热力图
            entity_types = ['drugs', 'genes', 'pathways']
            top_n = 10
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for idx, entity_type in enumerate(entity_types):
                if entity_type in stats and stats[entity_type]['top_20']:
                    entities, counts = zip(*stats[entity_type]['top_20'][:top_n])
                    axes[idx].barh(range(len(entities)), counts[::-1])
                    axes[idx].set_yticks(range(len(entities)))
                    axes[idx].set_yticklabels(entities[::-1])
                    axes[idx].set_xlabel('提及次数')
                    axes[idx].set_title(f'Top {top_n} {entity_type}')
            
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"{output_prefix}_entity_frequencies.png", dpi=300)
            plt.close()
            
            logger.info("✅ 可视化图表已生成")
            
        except Exception as e:
            logger.warning(f"创建可视化图表时出错: {e}")

# 简化接口
def extract_entities_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """简化接口，用于兼容旧代码"""
    extractor = EnhancedEntityExtractor(use_spacy=True)
    return extractor.analyze_dataframe(df)

def get_entity_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """简化接口，用于兼容旧代码"""
    extractor = EnhancedEntityExtractor(use_spacy=True)
    return extractor.get_entity_statistics(df)