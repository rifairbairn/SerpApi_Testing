from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

SCORE_CACHE_COLUMNS = [
    "URL",
    "CompanyName",
    "GPT_Subject",
    "GPT_Mentioned",
    "GPT_Relevance",
    "GPT_Usefulness",
    "ScoredAt",
]


def _make_key(url: str, company_name: str) -> str:
    return f"{url.strip().lower()}|||{company_name.strip().lower()}"


def load_score_cache(cache_path: str | Path) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.exists():
        return pd.DataFrame(columns=SCORE_CACHE_COLUMNS)
    df = pd.read_csv(path)
    for col in SCORE_CACHE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[SCORE_CACHE_COLUMNS]


def save_score_cache(df: pd.DataFrame, cache_path: str | Path) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _key_series(df: pd.DataFrame) -> pd.Series:
    return (
        df["URL"].fillna("").str.strip().str.lower()
        + "|||"
        + df["CompanyName"].fillna("").str.strip().str.lower()
    )


def get_cached_score(
    cache_df: pd.DataFrame, url: str, company_name: str
) -> Optional[Dict]:
    if cache_df.empty or not url:
        return None
    key = _make_key(url, company_name)
    rows = cache_df[_key_series(cache_df) == key]
    if rows.empty:
        return None
    row = rows.iloc[-1]
    return {
        "subject": row["GPT_Subject"],
        "mentioned": row["GPT_Mentioned"],
        "relevance": row["GPT_Relevance"],
        "usefulness": row["GPT_Usefulness"],
    }


def upsert_score_cache(
    cache_df: pd.DataFrame,
    url: str,
    company_name: str,
    score: Dict,
) -> pd.DataFrame:
    timestamp = pd.Timestamp.utcnow().isoformat()
    new_row = {
        "URL": url.strip(),
        "CompanyName": company_name.strip(),
        "GPT_Subject": score.get("subject"),
        "GPT_Mentioned": score.get("mentioned"),
        "GPT_Relevance": score.get("relevance"),
        "GPT_Usefulness": score.get("usefulness"),
        "ScoredAt": timestamp,
    }

    if not cache_df.empty:
        key = _make_key(url, company_name)
        mask = _key_series(cache_df) == key
        if mask.any():
            cache_df = cache_df.copy()
            for k, v in new_row.items():
                cache_df.loc[mask, k] = v
            return cache_df[SCORE_CACHE_COLUMNS]

    return pd.concat(
        [cache_df, pd.DataFrame([new_row], columns=SCORE_CACHE_COLUMNS)],
        ignore_index=True,
    )[SCORE_CACHE_COLUMNS]
