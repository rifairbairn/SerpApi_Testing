"""
Strategy test harness.

Runs every name-strategy x search-strategy combination for a sample of companies,
fetches the top N results from SerpAPI, and scores each result with ChatGPT.

Resumable:
  - Completed queries (Sedol + query string) are skipped if already present in the
    output CSV.
  - Scored articles (URL + company name) are skipped via the article score cache so
    the same article brought back by multiple strategies is only sent to ChatGPT once.
  - ChatGPT name suggestions are cached per company via the existing name cache.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd

from article_score_cache import (
    get_cached_score,
    load_score_cache,
    save_score_cache,
    upsert_score_cache,
)
from chatgpt_handler import ChatGPTAnalyser, infer_target_strategy_type
from database_handler import DatabaseHandler
from query_strategies import build_strategy_queries, ALL_SEARCH_STRATEGIES
from search_name_cache import (
    get_cached_query_records,
    load_cache,
    save_cache,
    upsert_query_cache_row,
)
from serp_api_handler import run_news_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE = "RothkoFO"
TICKER = "MS664220"
MAX_COMPANIES = 50
MAX_GPT_TARGET_CANDIDATES = 5
TOP_N_RESULTS = 25          # results to score per query (1 credit = 100 results fetched)
SEARCH_ENGINE = "google"
RANDOM_STATE = 42

# Domains that produce auto-generated / templated content.
# Articles from these domains are assigned a fixed low score without calling GPT.
DOMAIN_LOW_SCORE: Dict[str, Dict[str, Any]] = {
    "simplywall.st": {"subject": "No", "mentioned": "Yes", "relevance": 5, "usefulness": 0},
}

# TEST_MODE controls which queries are generated per company:
#   "target_comparison" -- all target types x corporate_actions only
#                          (tests name formulation quality, level playing field)
#   "search_comparison" -- official_unquoted only x all search strategies
#                          (tests search modifier quality, level playing field)
#   "production"        -- best-known routing per target type
TEST_MODE = "target_comparison"

POS_DATE = (pd.Timestamp.today() - pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
SEARCH_START_DATE = (pd.Timestamp.today() - pd.offsets.MonthEnd(5)).strftime("%m/%d/%Y")
SEARCH_END_DATE = pd.Timestamp.today().strftime("%m/%d/%Y")

OUTPUT_FILENAME = f"test_run_output_{TEST_MODE}.csv"
NAME_CACHE_FILENAME = "chatgpt_company_target_cache.csv"
SCORE_CACHE_FILENAME = "article_score_cache.csv"

OUTPUT_COLUMNS = [
    "Sedol",
    "Company Name",
    "Query Rank",
    "Query",
    "Target Query",
    "Target Strategy Type",
    "Search Strategy",
    "Result Rank",
    "Total Results",
    "Title",
    "Snippet",
    "Source",
    "Link",
    "Published",
    "GPT_Subject",
    "GPT_Mentioned",
    "GPT_Relevance",
    "GPT_Usefulness",
    "Search Error",
    "Score Error",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _extract(result: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        val = result.get(key)
        if isinstance(val, str):
            cleaned = val.strip()
            if cleaned:
                return cleaned
        elif val is not None:
            return str(val)
    return None


def _quote(value: str) -> str:
    cleaned = " ".join(_safe_str(value).split())
    if not cleaned:
        return ""
    if cleaned.startswith('"') and cleaned.endswith('"'):
        return cleaned
    return f'"{ cleaned.replace(chr(34), "")}"'


def _dedupe_candidates(
    candidates: List[Dict[str, str]],
    company_name: str,
    existing_names: List[str],
) -> List[Dict[str, str]]:
    seen: Set[str] = set()
    result = []
    existing_text = "; ".join(existing_names)
    for item in candidates:
        query = " ".join(_safe_str(item.get("query")).split())
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        strategy_type = _safe_str(item.get("strategy_type") or item.get("strategy"))
        if not strategy_type or strategy_type in {"chatgpt_target", "legacy_brand_name"}:
            strategy_type = infer_target_strategy_type(query, company_name, existing_text)
        result.append({"query": query, "strategy_type": strategy_type})
    return result


def _split_names(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [_safe_str(x) for x in raw if _safe_str(x)]
    text = _safe_str(raw)
    if not text:
        return []
    return [p.strip() for p in text.split(";") if p.strip()]


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def _load_completed_queries(output_path: Path) -> FrozenSet[Tuple[str, str]]:
    """Return the set of (sedol, query) pairs that are already in the output file."""
    if not output_path.exists():
        return frozenset()
    try:
        df = pd.read_csv(output_path, usecols=["Sedol", "Query"], dtype=str)
        return frozenset(
            zip(df["Sedol"].fillna(""), df["Query"].fillna(""))
        )
    except Exception as exc:
        logging.warning("Could not read completed queries from %s: %s", output_path, exc)
        return frozenset()


def _init_output_file(output_path: Path) -> None:
    """Write the CSV header if the file does not yet exist."""
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()


def _append_rows(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Name strategy builder
# ---------------------------------------------------------------------------


def _build_candidates(
    company_row: pd.Series,
    analyser: ChatGPTAnalyser,
    name_cache: pd.DataFrame,
) -> Tuple[List[Dict[str, str]], pd.DataFrame, bool]:
    sedol = _safe_str(company_row.get("Sedol"))
    company_name = _safe_str(company_row.get("EntityName"))
    existing_names = _split_names(company_row.get("EntitySearchNames"))

    candidates: List[Dict[str, str]] = []
    if company_name:
        candidates.append({"query": _quote(company_name), "strategy_type": "official_exact_quote"})
        candidates.append({"query": company_name, "strategy_type": "official_unquoted"})
    for name in existing_names:
        candidates.append({"query": name, "strategy_type": "existing_alias"})

    cache_updated = False
    if company_name:
        cached = get_cached_query_records(name_cache, sedol)
        if cached:
            logging.info("  Using %d cached name strategies for %s.", len(cached), company_name)
            candidates.extend(cached)
        else:
            try:
                suggestions = analyser.suggest_company_target_queries(
                    company_name,
                    existing_names="; ".join(existing_names),
                    max_names=MAX_GPT_TARGET_CANDIDATES,
                )
                candidates.extend(suggestions)
                if sedol and suggestions:
                    name_cache = upsert_query_cache_row(
                        name_cache,
                        entity_id=sedol,
                        entity_name=company_name,
                        search_queries=suggestions,
                    )
                    cache_updated = True
            except Exception as exc:
                logging.warning("  Could not generate name strategies for %s: %s", company_name, exc)

    return _dedupe_candidates(candidates, company_name, existing_names), name_cache, cache_updated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(os.getenv("OUTPUT_DIR", str(script_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / OUTPUT_FILENAME
    name_cache_path = output_dir / NAME_CACHE_FILENAME
    score_cache_path = output_dir / SCORE_CACHE_FILENAME

    _init_output_file(output_path)
    completed_queries = _load_completed_queries(output_path)
    logging.info("Resuming - %d query/company pairs already in output.", len(completed_queries))

    name_cache = load_cache(name_cache_path)
    score_cache = load_score_cache(score_cache_path)

    db = DatabaseHandler(DATABASE)
    analyser = ChatGPTAnalyser(model="gpt-4o-mini", scoring_model="gpt-5-mini")

    positions = db.get_index_positions(TICKER, POS_DATE)
    if positions is None or positions.empty:
        logging.error("No positions found for ticker %s on %s.", TICKER, POS_DATE)
        return 1

    positions = positions.sort_values(
        by=["EntityName", "Sedol"], na_position="last"
    ).reset_index(drop=True)
    sample_size = min(MAX_COMPANIES, len(positions))
    sampled = positions.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    logging.info(
        "Loaded %d positions, sampled %d companies. Date range: %s to %s.",
        len(positions), sample_size, SEARCH_START_DATE, SEARCH_END_DATE,
    )

    for company_idx, (_, row) in enumerate(sampled.iterrows(), start=1):
        sedol = _safe_str(row.get("Sedol"))
        company_name = _safe_str(row.get("EntityName"))

        if not company_name:
            logging.warning("Skipping company %d - EntityName is missing.", company_idx)
            continue

        logging.info("--- Company %d/%d: %s ---", company_idx, sample_size, company_name)

        candidates, name_cache, cache_updated = _build_candidates(row, analyser, name_cache)
        if cache_updated:
            save_cache(name_cache, name_cache_path)

        if not candidates:
            logging.warning("  No name strategies for %s - skipping.", company_name)
            continue

        queries = build_strategy_queries(candidates, mode=TEST_MODE)
        logging.info("  %d strategy queries generated.", len(queries))

        company_rows: List[Dict[str, Any]] = []

        for query_rank, sq in enumerate(queries, start=1):
            query_key = (sedol, sq.query)
            if query_key in completed_queries:
                logging.info("  [skip] Query %d already completed: %s", query_rank, sq.query)
                continue

            logging.info("  Query %d/%d: %s", query_rank, len(queries), sq.query)

            # --- SerpAPI ---
            try:
                news_results = run_news_search(
                    query=sq.query,
                    search_engine=SEARCH_ENGINE,
                    search_start_date=SEARCH_START_DATE,
                    search_end_date=SEARCH_END_DATE,
                )
            except Exception as exc:
                logging.error("  SerpAPI error: %s", exc)
                company_rows.append(
                    {
                        "Sedol": sedol,
                        "Company Name": company_name,
                        "Query Rank": query_rank,
                        "Query": sq.query,
                        "Target Query": sq.target_query,
                        "Target Strategy Type": sq.target_strategy_type,
                        "Search Strategy": sq.search_strategy,
                        "Result Rank": None,
                        "Total Results": None,
                        "Search Error": str(exc),
                    }
                )
                continue

            total = len(news_results)
            if total == 0:
                company_rows.append(
                    {
                        "Sedol": sedol,
                        "Company Name": company_name,
                        "Query Rank": query_rank,
                        "Query": sq.query,
                        "Target Query": sq.target_query,
                        "Target Strategy Type": sq.target_strategy_type,
                        "Search Strategy": sq.search_strategy,
                        "Result Rank": None,
                        "Total Results": 0,
                        "Search Error": None,
                    }
                )
                continue

            # --- Score top N results ---
            for result_rank, result in enumerate(news_results[:TOP_N_RESULTS], start=1):
                title = _extract(result, ["title", "headline", "name"])
                snippet = _extract(result, ["snippet", "description", "summary"])
                source = _extract(result, ["source", "publisher", "source_name"])
                link = _extract(result, ["link", "url"])
                published = _extract(result, ["date", "published", "published_date"])

                score = None
                score_error = None

                # Check domain blocklist before hitting GPT or cache
                if link:
                    _domain = link.split("/")[2].lower().replace("www.", "") if "//" in link else ""
                    _blocked = DOMAIN_LOW_SCORE.get(_domain)
                    if _blocked:
                        score = _blocked
                        logging.info("    [blocked domain: %s] %s", _domain, link)

                if score is None and link:
                    score = get_cached_score(score_cache, link, company_name)
                    if score:
                        logging.info("    [cached] %s", link)

                if score is None and (title or snippet):
                    try:
                        score = analyser.score_article(
                            company=company_name,
                            title=title or "",
                            snippet=snippet or "",
                        )
                        if score and link:
                            score_cache = upsert_score_cache(
                                score_cache, link, company_name, score
                            )
                    except Exception as exc:
                        score_error = str(exc)
                        logging.warning("    GPT score error: %s", exc)

                company_rows.append(
                    {
                        "Sedol": sedol,
                        "Company Name": company_name,
                        "Query Rank": query_rank,
                        "Query": sq.query,
                        "Target Query": sq.target_query,
                        "Target Strategy Type": sq.target_strategy_type,
                        "Search Strategy": sq.search_strategy,
                        "Result Rank": result_rank,
                        "Total Results": total,
                        "Title": title,
                        "Snippet": snippet,
                        "Source": source,
                        "Link": link,
                        "Published": published,
                        "GPT_Subject": score.get("subject") if isinstance(score, dict) else None,
                        "GPT_Mentioned": score.get("mentioned") if isinstance(score, dict) else None,
                        "GPT_Relevance": score.get("relevance") if isinstance(score, dict) else None,
                        "GPT_Usefulness": score.get("usefulness") if isinstance(score, dict) else None,
                        "Search Error": None,
                        "Score Error": score_error,
                    }
                )

        # Persist after each company to keep progress safe
        _append_rows(output_path, company_rows)
        save_score_cache(score_cache, score_cache_path)
        logging.info(
            "  Wrote %d rows for %s. Output: %s", len(company_rows), company_name, output_path
        )

    logging.info("Run complete. Output: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
