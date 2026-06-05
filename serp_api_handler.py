import logging
import os
from datetime import datetime

from dotenv import load_dotenv

"""
Need to add this to 'requests' library in 'utils.py' within 'select_proxy' function
proxies = proxies or {"http":"http://lon3.sme.zscaler.net:443",
                      "https":"http://lon3.sme.zscaler.net:443"}                    
"""

def run_news_search(query: str,
                    search_engine: str,
                    search_start_date: str,
                    search_end_date: str) -> list:

    if not query:
        raise ValueError("Query is empty")

    if not search_engine:
        raise ValueError("Specify search engine")

    try:
        start_dt = datetime.strptime(search_start_date, "%m/%d/%Y")
        end_dt = datetime.strptime(search_end_date, "%m/%d/%Y")
    except ValueError:
        raise TypeError("Enter valid search start and search end dates in MM/DD/YYYY format")

    if start_dt > end_dt:
        raise ValueError("Search start date cannot be after search end date")

    custom_date_range = f"cdr:1,cd_min:{search_start_date},cd_max:{search_end_date},sbd:1"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(script_dir, ".env"))
    api_key = os.getenv("SERPAPI_API_KEY")

    if not api_key:
        raise ValueError("SERPAPI_API_KEY is not set in .env")

    try:
        from serpapi import GoogleSearch
    except ImportError as exc:
        raise ImportError("Install google-search-results to run SerpAPI searches") from exc

    params = {
        "api_key": api_key,
        "engine": search_engine,
        "q": query,
        "tbs": custom_date_range,
        "tbm": "nws",
        "num": 100,
        "no_cache": "true",
        "filter": "0",
    }

    client = GoogleSearch(params)
    results = client.get_json()

    if "error" in results:
        raise RuntimeError(f"SerpAPI error: {results['error']}")

    search_info = results.get("search_information", {})
    search_status = search_info.get("news_results_state")
    news_results = results.get("news_results", [])

    if not news_results:
        logging.info("No news results for query '%s' (status: %s)", query, search_status)

    return news_results
