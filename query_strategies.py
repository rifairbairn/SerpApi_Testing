from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# ---------------------------------------------------------------------------
# Term lists
# ---------------------------------------------------------------------------

TERM_SETS = {
    "corporate_actions": [
        # Earnings & results
        "earnings", "results", "profit", "revenue",
        # Distributions
        "dividend", "buyback", "repurchase",
        # M&A & structure
        "merger", "acquisition", "takeover", "spin-off", "demerger", "restructuring",
        # Capital markets
        "offering", "rights issue", "bond", "debt issuance", "listing",
        # Legal & regulatory
        "lawsuit", "fine", "investigation", "penalty",
    ],
    "analyst_coverage": [
        "analyst", "rating", "price target", "recommendation",
        "forecast", "upgrade", "downgrade", "buy", "sell", "hold", "outperform",
    ],
    "management_change": [
        "CEO", "CFO", "chairman", "appointed", "resigned",
        "director", "board", "management", "executive",
    ],
    "exchange_filings": [
        "annual report", "quarterly results", "press release",
        "material fact", "regulatory filing", "prospectus",
    ],
}

NOISE_EXCLUSIONS = (
    "-site:instagram.com -site:youtube.com -site:reddit.com "
    "-site:tiktok.com -site:twitter.com -site:x.com"
)

SOURCE_WHITELIST = (
    "site:reuters.com OR site:bloomberg.com OR site:ft.com OR site:wsj.com "
    "OR site:cnbc.com OR site:barrons.com OR site:prnewswire.com OR site:businesswire.com"
)

PLACEHOLDERS = {
    "corporate_actions": " OR ".join(f'"{t}"' for t in TERM_SETS["corporate_actions"]),
    "analyst_coverage":  " OR ".join(f'"{t}"' for t in TERM_SETS["analyst_coverage"]),
    "management_change": " OR ".join(f'"{t}"' for t in TERM_SETS["management_change"]),
    "exchange_filings":  " OR ".join(f'"{t}"' for t in TERM_SETS["exchange_filings"]),
    "noise_exclusions":  NOISE_EXCLUSIONS,
    "source_whitelist":  SOURCE_WHITELIST,
}

# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------

SEARCH_STRATEGIES = {
    # Baselines
    "broad_news":        "{target}",
    "announced":         "{target} announced",
    "noise_filtered":    "{target} {noise_exclusions}",
    # Term-filtered
    "corporate_actions": "{target} ({corporate_actions})",
    "analyst_coverage":  "{target} ({analyst_coverage})",
    "management_change": "{target} ({management_change})",
    "exchange_filings":  "{target} ({exchange_filings})",
    # Site-filtered (kept for comparison)
    "source_whitelist":  "{target} ({source_whitelist})",
}

# All search strategies available for Test B
ALL_SEARCH_STRATEGIES = list(SEARCH_STRATEGIES.keys())

# ---------------------------------------------------------------------------
# Routing tables
# ---------------------------------------------------------------------------

# TEST_MODE = "engine_comparison"
# official_unquoted only x broad_news only — isolates engine as the sole variable.
ENGINE_COMPARISON_TARGET = "official_unquoted"
ENGINE_COMPARISON_SEARCH = ["broad_news"]

# TEST_MODE = "target_comparison"
# Every target type gets the same single search strategy so name formulations
# are compared on a level playing field.
TARGET_COMPARISON_SEARCH = ["broad_news"]

# TEST_MODE = "search_comparison"
# official_unquoted only, paired with every search strategy so modifiers are
# compared on a level playing field.
SEARCH_COMPARISON_TARGET = "official_unquoted"

# TEST_MODE = "production"
# Best-known routing based on test results.
PRODUCTION_ROUTING: Dict[str, List[str]] = {
    "official_unquoted":             ["corporate_actions", "announced", "exchange_filings"],
    "partial_quote_disambiguation":  ["corporate_actions", "announced", "noise_filtered"],
    "existing_alias":                ["corporate_actions", "announced", "broad_news"],
    "official_exact_quote":          ["corporate_actions", "announced"],
    "short_common_name":             ["corporate_actions", "analyst_coverage", "noise_filtered"],
    "ticker_exchange":               ["broad_news", "corporate_actions", "announced"],
    "local_language_name":           ["broad_news", "announced"],
    "former_name":                   ["broad_news", "corporate_actions"],
    "abbreviation_acronym":          ["corporate_actions", "noise_filtered"],
    # legacy fallback
    "primary_entity_name":           ["corporate_actions", "announced", "exchange_filings"],
}

DEFAULT_SEARCH_STRATEGIES = ["corporate_actions", "announced", "broad_news"]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyQuery:
    query: str
    search_strategy: str
    target_query: str
    target_strategy_type: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalise_strategy_type(value: object) -> str:
    cleaned = _clean_text(value).lower().replace(" ", "_")
    return cleaned or "unknown_target"


def _build_query(target: str, search_strategy: str) -> str:
    template = SEARCH_STRATEGIES[search_strategy]
    return template.format(target=target, **PLACEHOLDERS)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_strategy_queries(
    company_targets: List[Dict[str, str]],
    mode: str = "production",
) -> List[StrategyQuery]:
    """
    Build the full list of (target x search strategy) queries for one company.

    mode:
      "target_comparison" -- all target types x corporate_actions only.
                             Tests name formulation quality on a level playing field.
      "search_comparison" -- official_unquoted only x all search strategies.
                             Tests search modifier quality on a level playing field.
      "production"        -- best-known routing per target type.
    """
    queries: List[StrategyQuery] = []

    for target in company_targets:
        target_query = _clean_text(target.get("query"))
        if not target_query:
            continue

        target_strategy_type = _normalise_strategy_type(
            target.get("strategy_type") or target.get("strategy")
        )

        if mode == "engine_comparison":
            if target_strategy_type != ENGINE_COMPARISON_TARGET:
                continue
            search_strategies = ENGINE_COMPARISON_SEARCH

        elif mode == "target_comparison":
            search_strategies = TARGET_COMPARISON_SEARCH

        elif mode == "search_comparison":
            if target_strategy_type != SEARCH_COMPARISON_TARGET:
                continue
            search_strategies = ALL_SEARCH_STRATEGIES

        else:  # production
            search_strategies = PRODUCTION_ROUTING.get(
                target_strategy_type, DEFAULT_SEARCH_STRATEGIES
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

    # Deduplicate
    unique: List[StrategyQuery] = []
    seen: set = set()
    for item in queries:
        key = item.query.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique
