from __future__ import annotations

import logging
import os
from datetime import datetime

import requests
import urllib3
from dotenv import load_dotenv

"""
Need to add this to 'requests' library in 'utils.py' within 'select_proxy' function
proxies = proxies or {"http":"http://lon3.sme.zscaler.net:443",
                      "https":"http://lon3.sme.zscaler.net:443"}
"""

# Zscaler re-signs HTTPS traffic with its own CA, which Python doesn't trust by default.
# Patch requests.Session so the serpapi library skips SSL verification.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_original_request = requests.Session.request


def _unverified_request(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _original_request(self, method, url, **kwargs)


requests.Session.request = _unverified_request


def _load_api_key() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(script_dir, ".env"))
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY is not set in .env")
    return api_key


def _validate_dates(search_start_date: str, search_end_date: str) -> None:
    try:
        start_dt = datetime.strptime(search_start_date, "%m/%d/%Y")
        end_dt   = datetime.strptime(search_end_date,   "%m/%d/%Y")
    except ValueError:
        raise TypeError("Enter valid search start and search end dates in MM/DD/YYYY format")
    if start_dt > end_dt:
        raise ValueError("Search start date cannot be after search end date")


def _flatten_google_news_results(raw_results: list) -> list:
    """
    google_news can return story clusters with nested 'stories' lists.
    Flatten these into individual article dicts so downstream code sees
    a uniform list of articles.
    """
    articles = []
    for item in raw_results:
        stories = item.get("stories")
        if stories:
            for story in stories:
                articles.append(story)
        else:
            articles.append(item)
    return articles


def _run_google_nws(
    query: str,
    search_start_date: str,
    search_end_date: str,
    top_n: int,
) -> list:
    """
    engine=google + tbm=nws.  Returns max 10 results per page so we paginate
    until we have top_n results or results are exhausted.
    Each page costs 1 SerpAPI credit.
    """
    try:
        from serpapi import GoogleSearch
    except ImportError as exc:
        raise ImportError("Install google-search-results to run SerpAPI searches") from exc

    api_key          = _load_api_key()
    custom_date_range = f"cdr:1,cd_min:{search_start_date},cd_max:{search_end_date},sbd:1"
    collected: list  = []
    page_size        = 10   # google+nws hard limit per page

    for start in range(0, top_n, page_size):
        params = {
            "api_key":  api_key,
            "engine":   "google",
            "q":        query,
            "tbs":      custom_date_range,
            "tbm":      "nws",
            "num":      page_size,
            "start":    start,
            "no_cache": "true",
            "filter":   "0",
        }
        results = GoogleSearch(params).get_json()

        if "error" in results:
            raise RuntimeError(f"SerpAPI error: {results['error']}")

        page_articles = results.get("news_results", [])
        collected.extend(page_articles)

        if len(page_articles) < page_size:
            break   # no more results available

    if not collected:
        logging.info("No news results for query '%s' (google_nws)", query)

    return collected[:top_n]


def _run_google_news(
    query: str,
    search_start_date: str,
    search_end_date: str,
    top_n: int,
) -> list:
    """
    engine=google_news.  Returns up to 100 results in a single call (1 credit).
    Result clusters with nested 'stories' are flattened to individual articles.
    Date filtering uses the tbs parameter same as google_nws.
    """
    try:
        from serpapi import GoogleSearch
    except ImportError as exc:
        raise ImportError("Install google-search-results to run SerpAPI searches") from exc

    api_key           = _load_api_key()
    custom_date_range = f"cdr:1,cd_min:{search_start_date},cd_max:{search_end_date},sbd:1"

    params = {
        "api_key":  api_key,
        "engine":   "google_news",
        "q":        query,
        "tbs":      custom_date_range,
        "num":      100,
        "no_cache": "true",
        "filter":   "0",
    }
    results = GoogleSearch(params).get_json()

    if "error" in results:
        raise RuntimeError(f"SerpAPI error: {results['error']}")

    raw = results.get("news_results", [])

    # Debug - remove after inspection
    import json
    with open("raw_google_news_debug.json", "w", encoding="utf-8") as f:
        json.dump(raw[:3], f, indent=2, ensure_ascii=False)

    articles = _flatten_google_news_results(raw)

    if not articles:
        logging.info("No news results for query '%s' (google_news)", query)

    return articles[:top_n]


def run_news_search(
    query: str,
    search_engine: str,
    search_start_date: str,
    search_end_date: str,
    top_n: int = 50,
) -> list:
    """
    Unified news search entry point.

    search_engine:
      "google_nws"  — engine=google + tbm=nws, paginates to reach top_n
                      (costs top_n/10 credits per query)
      "google_news" — engine=google_news, single call up to 100 results
                      (costs 1 credit per query regardless of top_n)
    """
    if not query:
        raise ValueError("Query is empty")
    if not search_engine:
        raise ValueError("Specify search engine")

    _validate_dates(search_start_date, search_end_date)

    if search_engine == "google_news":
        return _run_google_news(query, search_start_date, search_end_date, top_n)
    elif search_engine == "google_nws":
        return _run_google_nws(query, search_start_date, search_end_date, top_n)
    else:
        raise ValueError(f"Unknown search_engine '{search_engine}'. Use 'google_news' or 'google_nws'.")
