from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


TERM_SETS = {
    "corporate_actions": [
        "earnings",
        "results",
        "dividend",
        "buyback",
        "repurchase",
        "merger",
        "acquisition",
        "offering",
    ],
    "market_context": [
        "shares",
        "stock",
        "profit",
        "revenue",
        "guidance",
        "outlook",
        "upgrade",
        "downgrade",
    ],
    "exchange_filings": [
        "annual report",
        "quarterly results",
        "press release",
        "material fact",
        "regulatory filing",
        "prospectus",
    ],
}

NOISE_EXCLUSIONS = "-site:instagram.com -site:youtube.com -site:reddit.com -site:tiktok.com -site:twitter.com -site:x.com"

PLACEHOLDERS = {
    "corporate_actions": " OR ".join(f'"{ term}"' for term in TERM_SETS["corporate_actions"]),
    "market_context":    " OR ".join(f'"{ term}"' for term in TERM_SETS["market_context"]),
    "exchange_filings":  " OR ".join(f'"{ term}"' for term in TERM_SETS["exchange_filings"]),
    "noise_exclusions":  NOISE_EXCLUSIONS,
}

# Easy-to-edit search layer. Target terms come from ChatGPT; these decide how to use them.
SEARCH_STRATEGIES = {
    "broad_news":        "{target}",
    "announced":         "{target} announced",
    "corporate_actions": "{target} ({corporate_actions})",
    "market_context":    "{target} ({market_context})",
    "exchange_filings":  "{target} ({exchange_filings})",
    "noise_filtered":    "{target} {noise_exclusions}",
}

# Route each target type to the search strategies most likely to yield signal.
# source_whitelist and pr_wires removed — poor results for EM companies.
# abbreviation_acronym removed from primary — too ambiguous, near-zero results.
TARGET_STRATEGY_SEARCH_STRATEGIES = {
    "official_unquoted":            ["corporate_actions", "announced", "exchange_filings"],
    "partial_quote_disambiguation":  ["corporate_actions", "announced", "noise_filtered"],
    "existing_alias":               ["corporate_actions", "announced", "broad_news"],
    "official_exact_quote":         ["corporate_actions", "announced"],
    "short_common_name":            ["corporate_actions", "market_context", "noise_filtered"],
    "ticker_exchange":              ["broad_news", "corporate_actions", "announced"],
    "local_language_name":          ["broad_news", "announced"],
    "former_name":                  ["broad_news", "corporate_actions"],
    # legacy / fallback
    "primary_entity_name":          ["corporate_actions", "announced", "exchange_filings"],
    "abbreviation_acronym":         ["corporate_actions", "noise_filtered"],
}

DEFAULT_SEARCH_STRATEGIES = ["corporate_actions", "announced", "broad_news"]


@dataclass(frozen=True)
class StrategyQuery:
    query: str
    search_strategy: str
    target_query: str
    target_strategy_type: str


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalise_strategy_type(value: object) -> str:
    cleaned = _clean_text(value).lower().replace(" ", "_")
    return cleaned or "unknown_target"


def _build_query(target: str, search_strategy: str) -> str:
    template = SEARCH_STRATEGIES[search_strategy]
    return template.format(target=target, **PLACEHOLDERS)


def build_strategy_queries(company_targets: List[Dict[str, str]]) -> List[StrategyQuery]:
    queries: List[StrategyQuery] = []

    for target in company_targets:
        target_query = _clean_text(target.get("query"))
        if not target_query:
            continue

        target_strategy_type = _normalise_strategy_type(
            target.get("strategy_type") or target.get("strategy")
        )
        search_strategies = TARGET_STRATEGY_SEARCH_STRATEGIES.get(
            target_strategy_type,
            DEFAULT_SEARCH_STRATEGIES,
        )

        for search_strategy in search_strategies:
            if search_strategy not in SEARCH_STRATEGIES:
                continue

            queries.append(
                StrategyQuery(
                    query=_build_query(target_query, search_strategy),
                    search_strategy=search_strategy,
                    target_query=target_query,
                    target_strategy_type=target_strategy_type,
                )
            )

    unique: List[StrategyQuery] = []
    seen = set()
    for item in queries:
        key = item.query.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique
