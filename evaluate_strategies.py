"""
Strategy evaluation script.

Reads test_run_output.csv and produces ranked summaries of:
  1. Target strategy type performance
  2. Search strategy performance
  3. Combined (name x search) heatmap
  4. Per-company hit rate
  5. Source quality
  6. Result rank decay
  7. Uniqueness -- what % of each strategy's useful results are exclusive to it

Usage:
  python evaluate_strategies.py                                  # reads test_run_output.csv
  python evaluate_strategies.py test_run_output_target_comparison.csv
  python evaluate_strategies.py test_run_output_search_comparison.csv

Metrics:
  - query_count       : unique queries run for this strategy group
  - result_count      : total scored article rows
  - results_per_q     : avg results returned per query
  - zero_result_rate  : % of queries returning 0 results from Google
  - avg_total_results : mean Google result count per query
  - hit_rate          : % GPT_Subject == Yes
  - useful_hit_rate   : % GPT_Relevance >= RELEVANCE_THRESHOLD
  - avg_relevance     : mean GPT_Relevance
  - avg_usefulness    : mean GPT_Usefulness
  - p75_relevance     : 75th-percentile relevance
  - unique_useful     : useful results not found by any other strategy (uniqueness view only)
  - unique_useful_pct : unique_useful as % of that strategy's useful results
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(SCRIPT_DIR)))

_input_name = sys.argv[1] if len(sys.argv) > 1 else "test_run_output.csv"
INPUT_FILE = OUTPUT_DIR / _input_name
OUTPUT_EXCEL = OUTPUT_DIR / (_input_name.replace(".csv", "_evaluated.xlsx"))

RELEVANCE_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)

    for col in ["GPT_Relevance", "GPT_Usefulness", "Result Rank", "Total Results", "Query Rank"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_subject"]    = df["GPT_Subject"].str.strip().str.lower() == "yes"
    df["is_mentioned"]  = df["GPT_Mentioned"].str.strip().str.lower() == "yes"
    df["is_useful_hit"] = df["GPT_Relevance"] >= RELEVANCE_THRESHOLD
    df["has_score"]     = df["GPT_Relevance"].notna()
    df["has_error"]     = df["Search Error"].notna() | df["Score Error"].notna()
    df["url_norm"]      = df["Link"].str.split("?").str[0].str.lower().str.strip()
    df["zero_results"]  = df["Total Results"].fillna(0) == 0

    return df


# ---------------------------------------------------------------------------
# Core metric builder
# ---------------------------------------------------------------------------

def _query_counts(df: pd.DataFrame, group_by: str | list[str]) -> pd.DataFrame:
    cols = [group_by] if isinstance(group_by, str) else group_by
    return (
        df.groupby(cols)["Query"]
        .nunique()
        .rename("query_count")
        .reset_index()
    )


def _result_volume(df: pd.DataFrame, group_by: str | list[str]) -> pd.DataFrame:
    """Per-query volume stats computed at query level to avoid double-counting."""
    cols = [group_by] if isinstance(group_by, str) else group_by
    query_level = df.drop_duplicates(subset=cols + ["Query"]).copy()
    g = query_level.groupby(cols)
    vol = g.agg(
        zero_result_rate  =("zero_results",  "mean"),
        avg_total_results =("Total Results", "mean"),
    ).reset_index()
    vol["zero_result_rate"]  = (vol["zero_result_rate"] * 100).round(1)
    vol["avg_total_results"] = vol["avg_total_results"].round(1)
    return vol


def build_metrics(df: pd.DataFrame, group_by: str | list[str]) -> pd.DataFrame:
    scored = df[df["has_score"]]
    g = scored.groupby(group_by)

    out = g.agg(
        result_count    =("GPT_Relevance", "count"),
        hit_rate        =("is_subject",    "mean"),
        useful_hit_rate =("is_useful_hit", "mean"),
        avg_relevance   =("GPT_Relevance", "mean"),
        avg_usefulness  =("GPT_Usefulness","mean"),
        p75_relevance   =("GPT_Relevance", lambda x: x.quantile(0.75)),
    ).reset_index()

    qc  = _query_counts(df, group_by)
    vol = _result_volume(df, group_by)
    out = out.merge(qc,  on=group_by, how="left")
    out = out.merge(vol, on=group_by, how="left")
    out["results_per_q"] = (out["result_count"] / out["query_count"]).round(1)

    out["hit_rate"]        = (out["hit_rate"]        * 100).round(1)
    out["useful_hit_rate"] = (out["useful_hit_rate"] * 100).round(1)
    out["avg_relevance"]   = out["avg_relevance"].round(1)
    out["avg_usefulness"]  = out["avg_usefulness"].round(1)
    out["p75_relevance"]   = out["p75_relevance"].round(1)

    cols = ([group_by] if isinstance(group_by, str) else group_by) + [
        "query_count", "result_count", "results_per_q",
        "zero_result_rate", "avg_total_results",
        "hit_rate", "useful_hit_rate", "avg_relevance", "avg_usefulness", "p75_relevance",
    ]
    return out[cols].sort_values("avg_relevance", ascending=False)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def view_target_strategy(df: pd.DataFrame) -> pd.DataFrame:
    return build_metrics(df, "Target Strategy Type")


def view_search_strategy(df: pd.DataFrame) -> pd.DataFrame:
    return build_metrics(df, "Search Strategy")


def view_combined(df: pd.DataFrame) -> pd.DataFrame:
    return build_metrics(df, ["Target Strategy Type", "Search Strategy"])


def view_per_company(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["has_score"]].copy()
    g = scored.groupby(["Sedol", "Company Name"])
    out = g.agg(
        result_count    =("GPT_Relevance", "count"),
        hit_rate        =("is_subject",    "mean"),
        useful_hit_rate =("is_useful_hit", "mean"),
        avg_relevance   =("GPT_Relevance", "mean"),
        avg_usefulness  =("GPT_Usefulness","mean"),
        p75_relevance   =("GPT_Relevance", lambda x: x.quantile(0.75)),
    ).reset_index()

    qc  = _query_counts(df, ["Sedol", "Company Name"])
    vol = _result_volume(df, ["Sedol", "Company Name"])
    out = out.merge(qc,  on=["Sedol", "Company Name"], how="left")
    out = out.merge(vol, on=["Sedol", "Company Name"], how="left")
    out["results_per_q"] = (out["result_count"] / out["query_count"]).round(1)

    out["hit_rate"]        = (out["hit_rate"]        * 100).round(1)
    out["useful_hit_rate"] = (out["useful_hit_rate"] * 100).round(1)
    out["avg_relevance"]   = out["avg_relevance"].round(1)
    out["avg_usefulness"]  = out["avg_usefulness"].round(1)
    out["p75_relevance"]   = out["p75_relevance"].round(1)

    return out[["Sedol", "Company Name", "query_count", "result_count", "results_per_q",
                "zero_result_rate", "avg_total_results",
                "hit_rate", "useful_hit_rate", "avg_relevance", "avg_usefulness", "p75_relevance"]
               ].sort_values("avg_relevance", ascending=False)


def view_source_quality(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["has_score"]].copy()
    scored["domain"] = scored["Link"].str.extract(r"https?://(?:www\.)?([^/]+)")
    g = scored.groupby("domain")
    out = g.agg(
        result_count    =("GPT_Relevance", "count"),
        hit_rate        =("is_subject",    "mean"),
        useful_hit_rate =("is_useful_hit", "mean"),
        avg_relevance   =("GPT_Relevance", "mean"),
        avg_usefulness  =("GPT_Usefulness","mean"),
        p75_relevance   =("GPT_Relevance", lambda x: x.quantile(0.75)),
    ).reset_index()
    out = out[out["result_count"] >= 5]
    out["hit_rate"]        = (out["hit_rate"]        * 100).round(1)
    out["useful_hit_rate"] = (out["useful_hit_rate"] * 100).round(1)
    out["avg_relevance"]   = out["avg_relevance"].round(1)
    out["avg_usefulness"]  = out["avg_usefulness"].round(1)
    out["p75_relevance"]   = out["p75_relevance"].round(1)
    return out.sort_values("avg_relevance", ascending=False)


def view_query_rank_decay(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["has_score"] & df["Result Rank"].notna()].copy()
    scored["Result Rank"] = scored["Result Rank"].astype(int)
    out = scored.groupby("Result Rank").agg(
        result_count  =("GPT_Relevance", "count"),
        avg_relevance =("GPT_Relevance", "mean"),
        avg_usefulness=("GPT_Usefulness","mean"),
        hit_rate      =("is_subject",    "mean"),
    ).reset_index()
    out["avg_relevance"]  = out["avg_relevance"].round(1)
    out["avg_usefulness"] = out["avg_usefulness"].round(1)
    out["hit_rate"]       = (out["hit_rate"] * 100).round(1)
    return out.sort_values("Result Rank")


def view_uniqueness(df: pd.DataFrame) -> pd.DataFrame:
    """For each target strategy, how many of its useful results are unique to it?"""
    scored = df[df["has_score"]].copy()
    useful = scored[scored["is_useful_hit"]].copy()

    url_strategies = (
        useful.groupby(["url_norm", "Company Name"])["Target Strategy Type"]
        .apply(set)
        .reset_index()
        .rename(columns={"Target Strategy Type": "found_by"})
    )
    useful = useful.merge(url_strategies, on=["url_norm", "Company Name"], how="left")
    useful["is_unique"] = useful["found_by"].apply(lambda s: len(s) == 1)

    rows = []
    for strat, grp in useful.groupby("Target Strategy Type"):
        total  = len(grp)
        unique = int(grp["is_unique"].sum())
        qcount = df[df["Target Strategy Type"] == strat]["Query"].nunique()
        rows.append({
            "Target Strategy Type": strat,
            "query_count":          qcount,
            "useful_results":       total,
            "unique_useful":        unique,
            "shared_useful":        total - unique,
            "unique_useful_pct":    round(unique / total * 100, 1) if total else 0,
            "unique_per_query":     round(unique / qcount, 2) if qcount else 0,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("unique_per_query", ascending=False)
        .reset_index(drop=True)
    )


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

    total_rows   = len(df)
    scored_rows  = df["has_score"].sum()
    error_rows   = df["has_error"].sum()
    companies    = df["Company Name"].nunique()
    uniq_queries = df["Query"].nunique()

    print(f"\nLoaded {total_rows:,} rows | {scored_rows:,} scored | "
          f"{error_rows} errors | {companies} companies | {uniq_queries} unique queries")

    views = {
        "Target Strategy":   view_target_strategy(df),
        "Search Strategy":   view_search_strategy(df),
        "Combined":          view_combined(df),
        "Per Company":       view_per_company(df),
        "Source Quality":    view_source_quality(df),
        "Result Rank Decay": view_query_rank_decay(df),
        "Uniqueness":        view_uniqueness(df),
    }

    for title, table in views.items():
        _print_table(title, table)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        for sheet_name, table in views.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            ws = writer.sheets[sheet_name[:31]]
            for col_cells in ws.columns:
                max_len = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in col_cells
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)

    print(f"\nExcel output: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
