# pip install sqlalchemy --proxy="http://lon3.sme.zscaler.net:443"

# test plan:
    # current searches
    # using chatgpt to finesse company search name
    # using news api
    # cutting down site list
    # removing site list and using search terms
    # using search terms and cut down site list

"""
Ideas:
"ChatGPT suggested name" + site list
"Company" earnings OR dividend OR buyback etc
"Company" site:prnewswire.com OR site:businesswire.com
"Company" announced
"Company" On google news only
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from chatgpt_handler import ChatGPTAnalyser, infer_target_strategy_type
from database_handler import DatabaseHandler
from query_strategies import build_strategy_queries
from search_name_cache import get_cached_query_records, load_cache, save_cache, upsert_query_cache_row
from serp_api_handler import run_news_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATABASE = "RothkoFO"
TICKER = "MS664220"
MAX_COMPANIES = 5
MAX_GPT_TARGET_CANDIDATES = 5
SEARCH_ENGINE = "google"
RANDOM_STATE = 42
POS_DATE = (pd.Timestamp.today() - pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
SEARCH_START_DATE = (pd.Timestamp.today() - pd.offsets.MonthEnd(5)).strftime("%m/%d/%Y")
SEARCH_END_DATE = pd.Timestamp.today().strftime("%m/%d/%Y")
OUTPUT_FILENAME = "search_term_test_output.csv"
SEARCH_TARGET_CACHE_FILENAME = "chatgpt_company_target_cache.csv"


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = " ".join(_safe_str(value).split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _query_candidate(query: str, strategy_type: str) -> Dict[str, str]:
    return {
        "query": " ".join(_safe_str(query).split()),
        "strategy_type": " ".join(_safe_str(strategy_type).split()) or "unspecified",
    }


def _quote_search_phrase(value: str) -> str:
    cleaned = " ".join(_safe_str(value).split())
    if not cleaned:
        return ""
    if cleaned.startswith('"') and cleaned.endswith('"'):
        return cleaned
    return f'"{cleaned.replace(chr(34), "")}"'


def _dedupe_query_candidates(
    values: List[Dict[str, str]],
    company_name: str,
    existing_names: List[str],
) -> List[Dict[str, str]]:
    seen = set()
    result = []
    existing_names_text = "; ".join(existing_names)
    for value in values:
        query = " ".join(_safe_str(value.get("query")).split())
        if not query:
            continue

        key = query.lower()
        if key in seen:
            continue
        seen.add(key)

        strategy_type = _safe_str(value.get("strategy_type") or value.get("strategy"))
        if strategy_type in {"", "chatgpt_target", "legacy_brand_name"}:
            strategy_type = infer_target_strategy_type(query, company_name, existing_names_text)

        result.append(_query_candidate(query, strategy_type))

    return result


def _split_search_names(raw_names: Any) -> List[str]:
    if raw_names is None:
        return []

    if isinstance(raw_names, (list, tuple, set)):
        parts = list(raw_names)
    else:
        text = _safe_str(raw_names)
        if not text:
            return []
        parts = [part.strip() for part in text.split(";")]

    return [part for part in (_safe_str(part) for part in parts) if part]


def _extract_field(result: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        value = result.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        elif value is not None:
            return str(value)
    return None


def _build_query_candidates(
    company_row: pd.Series,
    analyser: ChatGPTAnalyser,
    search_target_cache: pd.DataFrame,
) -> Tuple[List[Dict[str, str]], pd.DataFrame, bool]:
    company_id = _safe_str(company_row.get("Sedol"))
    company_name = _safe_str(company_row.get("EntityName"))
    existing_names = _split_search_names(company_row.get("EntitySearchNames"))

    candidates: List[Dict[str, str]] = []
    if company_name:
        candidates.append(_query_candidate(_quote_search_phrase(company_name), "official_exact_quote"))
        candidates.append(_query_candidate(company_name, "official_unquoted"))
    candidates.extend(_query_candidate(name, "existing_alias") for name in existing_names)

    cache_updated = False
    if company_name:
        cached_targets = get_cached_query_records(search_target_cache, company_id)
        if cached_targets:
            logging.info("Using %s cached AI target formulations for %s.", len(cached_targets), company_name)
            candidates.extend(cached_targets)
        else:
            try:
                suggestions = analyser.suggest_company_target_queries(
                    company_name,
                    existing_names="; ".join(existing_names),
                    max_names=MAX_GPT_TARGET_CANDIDATES,
                )
                candidates.extend(suggestions)

                if company_id and suggestions:
                    search_target_cache = upsert_query_cache_row(
                        search_target_cache,
                        entity_id=company_id,
                        entity_name=company_name,
                        search_queries=suggestions,
                    )
                    cache_updated = True
            except Exception as exc:
                logging.warning("Could not generate AI target formulations for %s: %s", company_name, exc)

    return _dedupe_query_candidates(candidates, company_name, existing_names), search_target_cache, cache_updated


def _analyse_top_result(
    analyser: ChatGPTAnalyser,
    company_name: str,
    query: str,
    search_strategy: str,
    target_query: str,
    target_strategy_type: str,
    result: Dict[str, Any],
    query_rank: int,
    result_count: int,
) -> Dict[str, Any]:
    title = _extract_field(result, ["title", "headline", "name"])
    snippet = _extract_field(result, ["snippet", "description", "summary"])
    source = _extract_field(result, ["source", "publisher", "source_name"])
    link = _extract_field(result, ["link", "url"])
    published = _extract_field(result, ["date", "published", "published_date"])

    analysis = None
    if title or snippet:
        analysis = analyser.analyse_article_relevance(company_name, title or "", snippet or "")

    return {
        "Company Name": company_name,
        "Query Rank": query_rank,
        "Query": query,
        "Target Query": target_query,
        "Target Strategy Type": target_strategy_type,
        "Search Strategy": search_strategy,
        "Result Count": result_count,
        "Top Title": title,
        "Top Snippet": snippet,
        "Top Source": source,
        "Top Link": link,
        "Top Published": published,
        "GPT_Subject": analysis.get("subject") if isinstance(analysis, dict) else None,
        "GPT_Mentioned": analysis.get("mentioned") if isinstance(analysis, dict) else None,
        "GPT_Relevancy": analysis.get("relevance") if isinstance(analysis, dict) else None,
        "GPT_Usefulness": analysis.get("usefulness") if isinstance(analysis, dict) else None,
    }


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_output_dir = script_dir
    output_dir = Path(os.getenv("OUTPUT_DIR", str(default_output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    search_target_cache_path = output_dir / SEARCH_TARGET_CACHE_FILENAME
    search_target_cache = load_cache(search_target_cache_path)

    db_handler = DatabaseHandler(DATABASE)
    analyser = ChatGPTAnalyser()

    positions = db_handler.get_index_positions(TICKER, POS_DATE)
    if positions is None or positions.empty:
        logging.error("No positions found for ticker %s.", TICKER)
        return 1

    # sort positions so sampling is consistent
    positions = positions.sort_values(by=["EntityName", "Sedol"], na_position="last").reset_index(drop=True)
    sample_size = min(MAX_COMPANIES, len(positions))
    sampled_positions = positions.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    logging.info("Loaded %s positions and sampled %s companies.", len(positions), sample_size)
    logging.info("Using date range %s to %s for SerpAPI searches.", SEARCH_START_DATE, SEARCH_END_DATE)

    rows: List[Dict[str, Any]] = []

    for idx, (_, company_row) in enumerate(sampled_positions.iterrows(), start=1):
        company_name = _safe_str(company_row.get("EntityName"))
        company_id = _safe_str(company_row.get("Sedol"))

        if not company_name:
            logging.warning("Skipping row %s because EntityName is missing.", idx)
            continue

        logging.info("Company %s/%s: %s", idx, sample_size, company_name)

        targets, search_target_cache, cache_updated = _build_query_candidates(company_row, analyser, search_target_cache)
        if cache_updated:
            save_cache(search_target_cache, search_target_cache_path)

        if not targets:
            logging.warning("No target formulations produced for %s.", company_name)
            continue

        queries = build_strategy_queries(targets)
        if not queries:
            logging.warning("No search strategy queries produced for %s.", company_name)
            continue

        for query_rank, strategy_query in enumerate(queries, start=1):
            query = strategy_query.query
            search_strategy = strategy_query.search_strategy
            target_query = strategy_query.target_query
            target_strategy_type = strategy_query.target_strategy_type

            try:
                news_results = run_news_search(
                    query=query,
                    search_engine=SEARCH_ENGINE,
                    search_start_date=SEARCH_START_DATE,
                    search_end_date=SEARCH_END_DATE,
                )
            except Exception as exc:
                logging.error("SerpAPI failed for %s with query '%s': %s", company_name, query, exc)
                rows.append(
                    {
                        "Sedol": company_id,
                        "Company Name": company_name,
                        "Query Rank": query_rank,
                        "Query": query,
                        "Target Query": target_query,
                        "Target Strategy Type": target_strategy_type,
                        "Search Strategy": search_strategy,
                        "Result Count": None,
                        "Top Title": None,
                        "Top Snippet": None,
                        "Top Source": None,
                        "Top Link": None,
                        "Top Published": None,
                        "GPT_Subject": None,
                        "GPT_Mentioned": None,
                        "GPT_Relevancy": None,
                        "GPT_Usefulness": None,
                        "Search Error": str(exc),
                    }
                )
                continue

            if not news_results:
                rows.append(
                    {
                        "Sedol": company_id,
                        "Company Name": company_name,
                        "Query Rank": query_rank,
                        "Query": query,
                        "Target Query": target_query,
                        "Target Strategy Type": target_strategy_type,
                        "Search Strategy": search_strategy,
                        "Result Count": 0,
                        "Top Title": None,
                        "Top Snippet": None,
                        "Top Source": None,
                        "Top Link": None,
                        "Top Published": None,
                        "GPT_Subject": None,
                        "GPT_Mentioned": None,
                        "GPT_Relevancy": None,
                        "GPT_Usefulness": None,
                        "Search Error": None,
                    }
                )
                continue

            rows.append(
                {
                    "Sedol": company_id,
                    "Company Name": company_name,
                    **_analyse_top_result(
                        analyser=analyser,
                        company_name=company_name,
                        query=query,
                        search_strategy=search_strategy,
                        target_query=target_query,
                        target_strategy_type=target_strategy_type,
                        result=news_results[0],
                        query_rank=query_rank,
                        result_count=len(news_results),
                    ),
                    "Search Error": None,
                }
            )

    output_df = pd.DataFrame(rows)
    if output_df.empty:
        logging.warning("No output rows were generated.")
        return 0

    output_path = output_dir / OUTPUT_FILENAME
    output_df.to_csv(output_path, index=False)

    logging.info("Wrote %s rows to %s", len(output_df), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
