# src/data_collection/pubmed_client_enhanced.py
import requests
import time
import xml.etree.ElementTree as ET
import json
import os
from tqdm import tqdm
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.settings_paddle import *

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PubMedEnhancedClient:
    """增强版PubMed客户端，支持大规模并行获取"""
    
    def __init__(self, email=EMAIL, tool=TOOL, max_workers=4):
        self.email = email
        self.tool = tool
        self.session = requests.Session()
        self.max_workers = max_workers
        self.session.headers.update({
            "User-Agent": f"{tool} ({email})"
        })
    
    def search_ids_comprehensive(self, query, max_results=10000, use_history=True):
        """综合搜索文献ID，支持大数量"""
        search_url = f"{PUBMED_BASE_URL}/esearch.fcgi"
        
        all_ids = []
        retstart = 0
        retmax = 10000  # PubMed单次最大
        
        with tqdm(total=max_results, desc="搜索文献ID") as pbar:
            while retstart < max_results:
                params = {
                    "db": "pubmed",
                    "term": query,
                    "retstart": retstart,
                    "retmax": min(retmax, max_results - retstart),
                    "retmode": "json",
                    "tool": self.tool,
                    "email": self.email
                }
                
                if use_history and retstart > 0:
                    # 使用WebEnv继续获取
                    pass
                
                try:
                    response = self.session.get(search_url, params=params, 
                                               timeout=REQUEST_TIMEOUT)
                    response.raise_for_status()
                    
                    data = response.json()
                    batch_ids = data.get("esearchresult", {}).get("idlist", [])
                    
                    if not batch_ids:
                        break
                    
                    all_ids.extend(batch_ids)
                    retstart += len(batch_ids)
                    pbar.update(len(batch_ids))
                    
                    logger.info(f"已获取 {len(all_ids)} 个ID")
                    
                    # 避免请求过快
                    time.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"搜索失败: {e}")
                    break
        
        logger.info(f"总计获取 {len(all_ids)} 个文献ID")
        return all_ids
    
    def fetch_parallel(self, id_list, batch_size=200, max_workers=None):
        """并行获取文献详情"""
        if max_workers is None:
            max_workers = self.max_workers
        
        all_articles = []
        
        # 分批处理
        batches = [id_list[i:i + batch_size] 
                  for i in range(0, len(id_list), batch_size)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._fetch_batch, batch): i 
                      for i, batch in enumerate(batches)}
            
            for future in tqdm(as_completed(futures), total=len(batches), 
                              desc="并行下载文献"):
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                except Exception as e:
                    logger.error(f"批次下载失败: {e}")
        
        return all_articles
    
    def _fetch_batch(self, batch_ids):
        """获取单批文献"""
        fetch_url = f"{PUBMED_BASE_URL}/efetch.fcgi"
        id_str = ",".join(batch_ids)
        
        params = {
            "db": "pubmed",
            "id": id_str,
            "retmode": "xml",
            "rettype": "abstract",
            "tool": self.tool,
            "email": self.email
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(fetch_url, params=params, 
                                          timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                articles = self._parse_articles_enhanced(root)
                return articles
                
            except Exception as e:
                logger.warning(f"批次获取失败，重试 {attempt + 1}/{MAX_RETRIES}: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
        
        return []
    
    def _parse_articles_enhanced(self, root):
        """增强版解析，包含更多字段"""
        articles = []
        
        for article in root.findall(".//PubmedArticle"):
            try:
                # 基础信息
                pmid = self._get_text(article, ".//PMID")
                title = self._get_text(article, ".//ArticleTitle")
                
                # 摘要（合并所有部分）
                abstract_parts = []
                for abstract_elem in article.findall(".//Abstract/AbstractText"):
                    text = abstract_elem.text
                    label = abstract_elem.get("Label")
                    if text:
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
                abstract = " ".join(abstract_parts)
                
                # 作者
                authors = []
                for author_elem in article.findall(".//Author"):
                    last_name = self._get_text(author_elem, "LastName")
                    fore_name = self._get_text(author_elem, "ForeName")
                    if last_name:
                        author_name = last_name
                        if fore_name:
                            author_name = f"{fore_name} {author_name}"
                        authors.append(author_name)
                
                # 期刊信息
                journal = self._get_text(article, ".//Journal/Title")
                volume = self._get_text(article, ".//Journal/JournalIssue/Volume")
                issue = self._get_text(article, ".//Journal/JournalIssue/Issue")
                pages = self._get_text(article, ".//Pagination/MedlinePgn")
                
                # 发表日期
                pub_date = self._parse_pub_date_enhanced(article)
                
                # MeSH术语
                mesh_terms = []
                for mesh_elem in article.findall(".//MeshHeading/DescriptorName"):
                    term = mesh_elem.text
                    if term:
                        mesh_terms.append(term)
                
                # 文章类型
                publication_types = []
                for type_elem in article.findall(".//PublicationType"):
                    if type_elem.text:
                        publication_types.append(type_elem.text)
                
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": "; ".join(authors),
                    "journal": journal,
                    "volume": volume,
                    "issue": issue,
                    "pages": pages,
                    "pub_date": pub_date,
                    "mesh_terms": "; ".join(mesh_terms),
                    "publication_types": "; ".join(publication_types),
                    "fetch_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "full_text": f"{title}. {abstract}"  # 用于分析的完整文本
                })
                
            except Exception as e:
                logger.warning(f"解析文章失败: {e}")
                continue
        
        return articles
    
    def _get_text(self, element, xpath):
        """安全获取文本"""
        elem = element.find(xpath)
        return elem.text if elem is not None else ""
    
    def _parse_pub_date_enhanced(self, article):
        """增强版日期解析"""
        pub_date_elem = article.find(".//PubDate")
        if pub_date_elem is None:
            return ""
        
        year = self._get_text(pub_date_elem, "Year")
        month = self._get_text(pub_date_elem, "Month")
        day = self._get_text(pub_date_elem, "Day")
        
        if year:
            date_parts = [year]
            if month:
                month_map = {"Jan": "01", "Feb": "02", "Mar": "03", 
                           "Apr": "04", "May": "05", "Jun": "06",
                           "Jul": "07", "Aug": "08", "Sep": "09",
                           "Oct": "10", "Nov": "11", "Dec": "12"}
                month = month_map.get(month, month.zfill(2))
                date_parts.append(month)
                if day:
                    date_parts.append(day.zfill(2))
            return "-".join(date_parts)
        
        return ""