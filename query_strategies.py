from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


HIGH_QUALITY_SITE_LIST = [
    "site:reuters.com",
    "site:bloomberg.com",
    "site:ft.com",
    "site:wsj.com",
    "site:cnbc.com",
    "site:barrons.com",
]

PR_WIRE_SITES = [
    "site:prnewswire.com",
    "site:businesswire.com",
]

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
}

PLACEHOLDERS = {
    "high_quality_sites": " OR ".join(HIGH_QUALITY_SITE_LIST),
    "pr_wire_sites": " OR ".join(PR_WIRE_SITES),
    "corporate_actions": " OR ".join(f'"{term}"' for term in TERM_SETS["corporate_actions"]),
    "market_context": " OR ".join(f'"{term}"' for term in TERM_SETS["market_context"]),
}

# Easy-to-edit search layer. Target terms come from ChatGPT; these decide how to use them.
SEARCH_STRATEGIES = {
    "broad_news": "{target}",
    "announced": "{target} announced",
    "source_whitelist": "{target} ({high_quality_sites} OR {pr_wire_sites})",
    "pr_wires": "{target} ({pr_wire_sites})",
    "corporate_actions": "{target} ({corporate_actions})",
    "market_context": "{target} ({market_context})",
}

# Route noisier target forms away from the broadest searches.
TARGET_STRATEGY_SEARCH_STRATEGIES = {
    "primary_entity_name": ["broad_news", "source_whitelist", "corporate_actions", "announced"],
    "existing_alias": ["broad_news", "source_whitelist", "corporate_actions"],
    "official_exact_quote": ["broad_news", "source_whitelist", "corporate_actions", "announced"],
    "official_unquoted": ["broad_news", "market_context", "announced"],
    "short_common_name": ["market_context", "corporate_actions", "announced"],
    "ticker_exchange": ["broad_news", "source_whitelist", "corporate_actions"],
    "abbreviation_acronym": ["source_whitelist", "market_context"],
    "partial_quote_disambiguation": ["broad_news", "corporate_actions", "announced"],
    "local_language_name": ["broad_news", "announced"],
}

DEFAULT_SEARCH_STRATEGIES = ["broad_news", "source_whitelist", "corporate_actions"]


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
