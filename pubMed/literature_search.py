"""
PubMed API wrapper using NCBI E-utilities.

Provides the PubMedSearcher class for searching PubMed and fetching article details.
"""
from __future__ import annotations

import requests
import xml.etree.ElementTree as ET
import time
from typing import Optional


class PubMedSearcher:
    """Search PubMed using the E-utilities API."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: str, api_key: Optional[str] = None):
        """
        Initialize the searcher.

        Args:
            email: Required by NCBI for API usage
            api_key: Optional API key for higher rate limits (10 req/sec vs 3 req/sec)
        """
        self.email = email
        self.api_key = api_key

    def _make_request(self, endpoint: str, params: dict) -> requests.Response:
        """Make a request to the E-utilities API."""
        params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Rate limiting: 3 requests/sec without API key, 10 with
        time.sleep(0.35 if self.api_key else 1.1)
        return response

    def search(
        self,
        query: str,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        max_results: int = 1000
    ) -> list[str]:
        """
        Search PubMed and return PMIDs.

        Args:
            query: Boolean search query
            min_date: Start date (YYYY/MM/DD or YYYY)
            max_date: End date (YYYY/MM/DD or YYYY)
            max_results: Maximum number of results to return

        Returns:
            List of PubMed IDs (PMIDs)
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "usehistory": "y",
            "retmode": "json"
        }

        if min_date:
            params["mindate"] = min_date
            params["datetype"] = "pdat"  # Publication date
        if max_date:
            params["maxdate"] = max_date
            params["datetype"] = "pdat"

        response = self._make_request("esearch.fcgi", params)
        data = response.json()

        pmids = data.get("esearchresult", {}).get("idlist", [])
        total_count = data.get("esearchresult", {}).get("count", 0)

        print(f"Found {total_count} results, retrieving {len(pmids)}")
        return pmids

    def fetch_details(self, pmids: list[str]) -> list[dict]:
        """
        Fetch article details for a list of PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of article dictionaries with metadata
        """
        if not pmids:
            return []

        articles = []
        batch_size = 100

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            print(f"Fetching details for articles {i+1}-{min(i+batch_size, len(pmids))}...")

            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml"
            }

            response = self._make_request("efetch.fcgi", params)
            articles.extend(self._parse_articles(response.text))

        return articles

    def _parse_articles(self, xml_text: str) -> list[dict]:
        """Parse XML response into article dictionaries."""
        articles = []
        root = ET.fromstring(xml_text)

        for article in root.findall(".//PubmedArticle"):
            try:
                # Extract PMID
                pmid = article.find(".//PMID").text

                # Extract title
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else ""

                # Extract abstract
                abstract_parts = article.findall(".//AbstractText")
                abstract = " ".join(
                    (part.text or "") for part in abstract_parts
                )

                # Extract authors
                authors = []
                for author in article.findall(".//Author"):
                    lastname = author.find("LastName")
                    forename = author.find("ForeName")
                    if lastname is not None:
                        name = lastname.text
                        if forename is not None:
                            name = f"{forename.text} {name}"
                        authors.append(name)

                # Extract journal
                journal_elem = article.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else ""

                # Extract publication year
                year_elem = article.find(".//PubDate/Year")
                year = year_elem.text if year_elem is not None else ""

                # Extract keywords
                keywords = []
                for kw in article.findall(".//Keyword"):
                    if kw.text:
                        keywords.append(kw.text)

                # Extract DOI
                doi = ""
                for article_id in article.findall(".//ArticleId"):
                    if article_id.get("IdType") == "doi":
                        doi = article_id.text
                        break

                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": "; ".join(authors),
                    "journal": journal,
                    "year": year,
                    "keywords": "; ".join(keywords),
                    "doi": doi,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })

            except Exception as e:
                print(f"Error parsing article: {e}")
                continue

        return articles


