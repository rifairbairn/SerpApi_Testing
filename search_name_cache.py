from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

CACHE_COLUMNS = ["EntityID", "EntityName", "SearchNames", "Source", "CreatedAt", "UpdatedAt"]


def _clean_names(names: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for name in names:
        if not isinstance(name, str):
            continue
        value = " ".join(name.split()).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return cleaned


def _clean_query_records(records: List[Any]) -> List[Dict[str, str]]:
    cleaned = []
    seen = set()
    for item in records:
        if isinstance(item, str):
            query = item
            strategy_type = "chatgpt_target"
        elif isinstance(item, dict):
            query = item.get("query")
            strategy_type = item.get("strategy_type") or item.get("strategy") or "chatgpt_target"
        else:
            continue

        if not isinstance(query, str):
            continue

        value = " ".join(query.split()).strip()
        if not value:
            continue

        key = value.lower()
        if key in seen:
            continue
        seen.add(key)

        if not isinstance(strategy_type, str):
            strategy_type = "chatgpt_target"

        cleaned.append(
            {
                "query": value,
                "strategy_type": " ".join(strategy_type.split()).strip() or "chatgpt_target",
            }
        )

    return cleaned


def load_cache(cache_path: str | Path) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    df = pd.read_csv(path)
    missing = [col for col in CACHE_COLUMNS if col not in df.columns]
    for col in missing:
        df[col] = None
    return df[CACHE_COLUMNS]


def save_cache(df: pd.DataFrame, cache_path: str | Path) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def get_cached_names(cache_df: pd.DataFrame, entity_id: str) -> List[str]:
    if cache_df.empty or not entity_id:
        return []

    rows = cache_df.loc[cache_df["EntityID"].astype(str) == str(entity_id)]
    if rows.empty:
        return []

    raw = rows.iloc[-1].get("SearchNames")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return _clean_names([str(item) for item in parsed])
    except Exception:
        pass

    return _clean_names([part.strip() for part in str(raw).split(";")])


def get_cached_query_records(cache_df: pd.DataFrame, entity_id: str) -> List[Dict[str, str]]:
    if cache_df.empty or not entity_id:
        return []

    rows = cache_df.loc[cache_df["EntityID"].astype(str) == str(entity_id)]
    if rows.empty:
        return []

    raw = rows.iloc[-1].get("SearchNames")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return _clean_query_records(parsed)
    except Exception:
        pass

    return _clean_query_records([part.strip() for part in str(raw).split(";")])


def upsert_cache_row(
    cache_df: pd.DataFrame,
    entity_id: str,
    entity_name: str,
    search_names: List[str],
    source: str = "chatgpt",
) -> pd.DataFrame:
    timestamp = pd.Timestamp.utcnow().isoformat()
    row = {
        "EntityID": entity_id,
        "EntityName": entity_name,
        "SearchNames": json.dumps(_clean_names(search_names), ensure_ascii=False),
        "Source": source,
        "CreatedAt": timestamp,
        "UpdatedAt": timestamp,
    }

    if cache_df.empty:
        return pd.DataFrame([row], columns=CACHE_COLUMNS)

    mask = cache_df["EntityID"].astype(str) == str(entity_id)
    if mask.any():
        cache_df = cache_df.copy()
        for key, value in row.items():
            cache_df.loc[mask, key] = value
        return cache_df[CACHE_COLUMNS]

    return pd.concat([cache_df, pd.DataFrame([row], columns=CACHE_COLUMNS)], ignore_index=True)[CACHE_COLUMNS]


def upsert_query_cache_row(
    cache_df: pd.DataFrame,
    entity_id: str,
    entity_name: str,
    search_queries: List[Dict[str, str]],
    source: str = "chatgpt_target",
) -> pd.DataFrame:
    timestamp = pd.Timestamp.utcnow().isoformat()
    row = {
        "EntityID": entity_id,
        "EntityName": entity_name,
        "SearchNames": json.dumps([item["query"] for item in _clean_query_records(search_queries)], ensure_ascii=False),
        "Source": source,
        "CreatedAt": timestamp,
        "UpdatedAt": timestamp,
    }

    if cache_df.empty:
        return pd.DataFrame([row], columns=CACHE_COLUMNS)

    mask = cache_df["EntityID"].astype(str) == str(entity_id)
    if mask.any():
        cache_df = cache_df.copy()
        created_at = cache_df.loc[mask, "CreatedAt"].iloc[-1]
        for key, value in row.items():
            cache_df.loc[mask, key] = value
        cache_df.loc[mask, "CreatedAt"] = created_at
        return cache_df[CACHE_COLUMNS]

    return pd.concat([cache_df, pd.DataFrame([row], columns=CACHE_COLUMNS)], ignore_index=True)[CACHE_COLUMNS]
