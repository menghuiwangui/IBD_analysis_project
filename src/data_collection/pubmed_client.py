# src/data_collection/pubmed_client.py
import requests
import time
import xml.etree.ElementTree as ET
import json
import os
from tqdm import tqdm
import logging
from pathlib import Path
from config.settings import *

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PubMedClient:
    def __init__(self, email=EMAIL, tool=TOOL):
        self.email = email
        self.tool = tool
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"{tool} ({email})"})
    
    def _make_request(self, url, params, retry_count=0):
        """通用请求函数"""
        try:
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if retry_count < MAX_RETRIES:
                logger.warning(f"Request failed, retrying ({retry_count + 1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY * (retry_count + 1))
                return self._make_request(url, params, retry_count + 1)
            else:
                logger.error(f"Request failed after {MAX_RETRIES} retries: {e}")
                raise
    
    def search_ids(self, query, max_results=10000):
        """搜索文献ID"""
        search_url = f"{PUBMED_BASE_URL}/esearch.fcgi"
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 100000),  # PubMed最大限制
            "retmode": "json",
            "tool": self.tool,
            "email": self.email
        }
        
        try:
            response = self._make_request(search_url, params)
            data = response.json()
            
            id_list = data.get("esearchresult", {}).get("idlist", [])
            count = int(data.get("esearchresult", {}).get("count", 0))
            
            logger.info(f"Found {count} articles for query: '{query}'")
            logger.info(f"Will fetch {len(id_list)} articles")
            
            return id_list
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return []
    
    def fetch_articles(self, id_list, batch_size=200):
        """批量获取文献详情"""
        fetch_url = f"{PUBMED_BASE_URL}/efetch.fcgi"
        all_articles = []
        
        if not id_list:
            return all_articles
        
        for i in tqdm(range(0, len(id_list), batch_size), desc="Downloading articles"):
            batch_ids = id_list[i:i + batch_size]
            id_str = ",".join(batch_ids)
            
            params = {
                "db": "pubmed",
                "id": id_str,
                "retmode": "xml",
                "tool": self.tool,
                "email": self.email
            }
            
            try:
                response = self._make_request(fetch_url, params)
                root = ET.fromstring(response.content)
                articles = self._parse_articles_xml(root)
                all_articles.extend(articles)
                
            except Exception as e:
                logger.error(f"Failed to fetch batch starting at {i}: {e}")
                continue
            
            # 礼貌性延迟
            time.sleep(0.3)
        
        return all_articles
    
    def _parse_articles_xml(self, root):
        """解析XML"""
        articles = []
        
        for article in root.findall(".//PubmedArticle"):
            try:
                # PMID
                pmid_elem = article.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""
                
                # 标题
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else ""
                
                # 摘要
                abstract_elems = article.findall(".//Abstract/AbstractText")
                abstract_parts = []
                for elem in abstract_elems:
                    if elem.text:
                        abstract_parts.append(elem.text)
                abstract = " ".join(abstract_parts)
                
                # 作者
                authors = []
                for author_elem in article.findall(".//Author"):
                    last_name = author_elem.find("LastName")
                    fore_name = author_elem.find("ForeName")
                    if last_name is not None and last_name.text:
                        author_name = last_name.text
                        if fore_name is not None and fore_name.text:
                            author_name = f"{fore_name.text} {author_name}"
                        authors.append(author_name)
                
                # 期刊
                journal_elem = article.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else ""
                
                # 发表日期
                pub_date = self._parse_pub_date(article)
                
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": "; ".join(authors),
                    "journal": journal,
                    "pub_date": pub_date,
                    "fetch_time": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except Exception as e:
                logger.warning(f"Error parsing article: {e}")
                continue
        
        return articles
    
    def _parse_pub_date(self, article):
        """解析发表日期"""
        pub_date_elem = article.find(".//PubDate")
        if pub_date_elem is None:
            return ""
        
        year_elem = pub_date_elem.find("Year")
        month_elem = pub_date_elem.find("Month")
        day_elem = pub_date_elem.find("Day")
        
        year = year_elem.text if year_elem is not None else ""
        month = month_elem.text if month_elem is not None else ""
        day = day_elem.text if day_elem is not None else ""
        
        if year and month and day:
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        elif year and month:
            return f"{year}-{month.zfill(2)}"
        elif year:
            return year
        return ""
    
    def save_data(self, articles, filename=None):
        """保存数据"""
        if not filename:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ibd_articles_{timestamp}.json"
        
        filepath = RAW_DATA_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(articles)} articles to {filepath}")
        return str(filepath)
    
    def crawl(self, query, max_results=1000):
        """主爬取函数"""
        logger.info(f"Starting PubMed crawl for: '{query}'")
        
        # 1. 搜索ID
        id_list = self.search_ids(query, max_results)
        
        if not id_list:
            logger.warning("No articles found")
            return None, []
        
        # 2. 获取详情
        articles = self.fetch_articles(id_list)
        
        if not articles:
            logger.warning("No articles downloaded")
            return None, []
        
        # 3. 保存
        filename = self.save_data(articles)
        
        return filename, articles