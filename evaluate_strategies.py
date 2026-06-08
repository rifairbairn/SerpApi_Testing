"""
Strategy evaluation script.

Reads test_run_output.csv and produces ranked summaries of:
  1. Target strategy type performance  (name formulation quality)
  2. Search strategy performance        (query modifier quality)
  3. Combined (name × search) heatmap
  4. Per-company hit rate               (which companies are hard to find)
  5. Source quality                     (which domains return useful results)

Metrics used throughout:
  - hit_rate       : % of result rows where GPT_Subject == "Yes"
  - avg_relevance  : mean GPT_Relevance across all rows
  - avg_usefulness : mean GPT_Usefulness across all rows
  - useful_hits    : % of rows where GPT_Relevance >= 50 (article focuses on company)
  - p75_relevance  : 75th-percentile relevance (robustness signal)
  - result_count   : total rows (volume signal)

Output: printed tables + evaluate_strategies_output.xlsx (one sheet per view).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(SCRIPT_DIR)))

INPUT_FILE = OUTPUT_DIR / "test_run_output.csv"
OUTPUT_EXCEL = OUTPUT_DIR / "evaluate_strategies_output.xlsx"

# Relevance threshold for "useful hit" — article is primarily about the company
RELEVANCE_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)

    # Numeric columns
    for col in ["GPT_Relevance", "GPT_Usefulness", "Result Rank", "Total Results", "Query Rank"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Boolean flags
    df["is_subject"] = df["GPT_Subject"].str.strip().str.lower() == "yes"
    df["is_mentioned"] = df["GPT_Mentioned"].str.strip().str.lower() == "yes"
    df["is_useful_hit"] = df["GPT_Relevance"] >= RELEVANCE_THRESHOLD
    df["has_score"] = df["GPT_Relevance"].notna()
    df["has_error"] = df["Search Error"].notna() | df["Score Error"].notna()

    return df


# ---------------------------------------------------------------------------
# Metric builder
# ---------------------------------------------------------------------------

AGG: dict[str, Any] = {
    "result_count": ("GPT_Relevance", "count"),
    "hit_rate":      ("is_subject",    "mean"),
    "useful_hit_rate": ("is_useful_hit", "mean"),
    "avg_relevance": ("GPT_Relevance", "mean"),
    "avg_usefulness": ("GPT_Usefulness", "mean"),
    "p75_relevance": ("GPT_Relevance", lambda x: x.quantile(0.75)),
}


def build_metrics(df: pd.DataFrame, group_by: str | list[str]) -> pd.DataFrame:
    scored = df[df["has_score"]]
    g = scored.groupby(group_by)
    out = g.agg(**AGG).reset_index()
    out["hit_rate"] = (out["hit_rate"] * 100).round(1)
    out["useful_hit_rate"] = (out["useful_hit_rate"] * 100).round(1)
    out["avg_relevance"] = out["avg_relevance"].round(1)
    out["avg_usefulness"] = out["avg_usefulness"].round(1)
    out["p75_relevance"] = out["p75_relevance"].round(1)
    return out.sort_values("avg_relevance", ascending=False)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def view_target_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Performance by name formulation type (Target Strategy Type)."""
    return build_metrics(df, "Target Strategy Type")


def view_search_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Performance by search modifier (broad_news, source_whitelist, etc.)."""
    return build_metrics(df, "Search Strategy")


def view_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Performance by (Target Strategy Type × Search Strategy) combination."""
    return build_metrics(df, ["Target Strategy Type", "Search Strategy"])


def view_per_company(df: pd.DataFrame) -> pd.DataFrame:
    """Per-company summary — identifies hard-to-find companies."""
    scored = df[df["has_score"]].copy()
    g = scored.groupby(["Sedol", "Company Name"])
    out = g.agg(**AGG).reset_index()
    out["hit_rate"] = (out["hit_rate"] * 100).round(1)
    out["useful_hit_rate"] = (out["useful_hit_rate"] * 100).round(1)
    out["avg_relevance"] = out["avg_relevance"].round(1)
    out["avg_usefulness"] = out["avg_usefulness"].round(1)
    out["p75_relevance"] = out["p75_relevance"].round(1)

    # Add query count per company
    qcount = (
        df.groupby(["Sedol", "Company Name"])["Query"]
        .nunique()
        .rename("unique_queries")
        .reset_index()
    )
    out = out.merge(qcount, on=["Sedol", "Company Name"], how="left")
    return out.sort_values("avg_relevance", ascending=False)


def view_source_quality(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Top domains by average relevance (min 5 articles scored)."""
    scored = df[df["has_score"]].copy()
    scored["domain"] = scored["Link"].str.extract(r"https?://(?:www\.)?([^/]+)")
    g = scored.groupby("domain")
    out = g.agg(**AGG).reset_index()
    out = out[out["result_count"] >= 5]
    out["hit_rate"] = (out["hit_rate"] * 100).round(1)
    out["useful_hit_rate"] = (out["useful_hit_rate"] * 100).round(1)
    out["avg_relevance"] = out["avg_relevance"].round(1)
    out["avg_usefulness"] = out["avg_usefulness"].round(1)
    out["p75_relevance"] = out["p75_relevance"].round(1)
    return out.sort_values("avg_relevance", ascending=False).head(top_n)


def view_query_rank_decay(df: pd.DataFrame) -> pd.DataFrame:
    """Average relevance by result rank position (1-25) — shows how fast quality drops."""
    scored = df[df["has_score"] & df["Result Rank"].notna()].copy()
    scored["Result Rank"] = scored["Result Rank"].astype(int)
    g = scored.groupby("Result Rank")
    out = g.agg(
        result_count=("GPT_Relevance", "count"),
        avg_relevance=("GPT_Relevance", "mean"),
        avg_usefulness=("GPT_Usefulness", "mean"),
        hit_rate=("is_subject", "mean"),
    ).reset_index()
    out["avg_relevance"] = out["avg_relevance"].round(1)
    out["avg_usefulness"] = out["avg_usefulness"].round(1)
    out["hit_rate"] = (out["hit_rate"] * 100).round(1)
    return out.sort_values("Result Rank")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    df = load_data(INPUT_FILE)

    total_rows = len(df)
    scored_rows = df["has_score"].sum()
    error_rows = df["has_error"].sum()
    companies = df["Company Name"].nunique()
    unique_queries = df["Query"].nunique()

    print(f"\nLoaded {total_rows:,} rows | {scored_rows:,} scored | "
          f"{error_rows} errors | {companies} companies | {unique_queries} unique queries")

    views = {
        "Target Strategy":     view_target_strategy(df),
        "Search Strategy":     view_search_strategy(df),
        "Combined":            view_combined(df),
        "Per Company":         view_per_company(df),
        "Source Quality":      view_source_quality(df),
        "Result Rank Decay":   view_query_rank_decay(df),
    }

    for title, table in views.items():
        _print_table(title, table)

    # Write Excel workbook
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        for sheet_name, table in views.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            ws = writer.sheets[sheet_name[:31]]
            # Auto-width columns
            for col_cells in ws.columns:
                max_len = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in col_cells
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)

    print(f"\nExcel output: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
