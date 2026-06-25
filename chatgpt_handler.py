from __future__ import annotations

import logging
import os
import re
import requests
from dotenv import load_dotenv
import json
from typing import Dict, List, Optional

# Load API key from the project .env file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(SCRIPT_DIR, ".env"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HTTP_PROXY = os.getenv("HTTP_PROXY")
HTTPS_PROXY = os.getenv("HTTPS_PROXY")

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _clean_text(value) -> str:
    return " ".join(str(value or "").split()).strip()


def _strip_outer_quotes(value: str) -> str:
    cleaned = _clean_text(value)
    if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
        return cleaned[1:-1].strip()
    return cleaned


def _normalise_for_compare(value: str) -> str:
    return _strip_outer_quotes(value).lower()


def _split_existing_names(existing_names: str) -> List[str]:
    return [_clean_text(part) for part in str(existing_names or "").split(";") if _clean_text(part)]


def _looks_like_ticker_exchange(query: str) -> bool:
    return bool(
        re.search(r"\b[A-Z0-9]{1,6}[.:]\s*[A-Z0-9]{1,10}\b", query)
        or re.search(r"\b[A-Z]{1,6}\s+(US|LN|HK|JP|GR|FP|SW|SS|SZ|KS|KL)\b", query)
    )


def _looks_like_acronym(query: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9&.-]{1,9}", query)) and not _looks_like_ticker_exchange(query)


def _infer_target_strategy_type(query: str, company_name: str, existing_names: str) -> str:
    cleaned_query = _clean_text(query)
    compare_names = [_normalise_for_compare(company_name)]
    compare_names.extend(_normalise_for_compare(name) for name in _split_existing_names(existing_names))
    compare_names = [name for name in compare_names if name]

    if cleaned_query.count('"') >= 2:
        if _normalise_for_compare(cleaned_query) in compare_names:
            return "official_exact_quote"
        return "partial_quote_disambiguation"

    if any(ord(char) > 127 for char in cleaned_query):
        return "local_language_name"

    if _looks_like_ticker_exchange(cleaned_query):
        return "ticker_exchange"

    if _looks_like_acronym(cleaned_query):
        return "abbreviation_acronym"

    if _normalise_for_compare(cleaned_query) in compare_names:
        return "official_unquoted"

    return "short_common_name"


def infer_target_strategy_type(query: str, company_name: str, existing_names: str = "") -> str:
    return _infer_target_strategy_type(query, company_name, existing_names)

class ChatGPTAnalyser:
    """
    This class uses OpenAI's ChatGPT API to analyze article titles and snippets.
    It determines if a company is the subject, assigns a relevance score, and measures usefulness.
    """

    def __init__(self, model="gpt-4o-mini", scoring_model: str | None = None):
        """
        Initializes the ChatGPTAnalyser class with API credentials and optional proxy settings.

        :param api_key: OpenAI API key.
        :param proxy: Proxy URL (e.g., "http://your.proxy.server:8080") or None.
        :param model: The OpenAI model to use for name suggestions.
        :param scoring_model: The OpenAI model to use for article scoring (defaults to model).
        """
        self.api_key = OPENAI_API_KEY
        self.model = model
        self.scoring_model = scoring_model or model

        # Configure proxy settings
        self.proxies = {
            key: value
            for key, value in {"http": HTTP_PROXY, "https": HTTPS_PROXY}.items()
            if value
        }

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("ChatGPTAnalyser initialized.")

    def _request_chat_completion(self, prompt, model: str | None = None):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model or self.model,
                    "messages": [{"role": "user", "content": prompt}]
                },
                proxies=self.proxies,
                timeout=15,
                verify=False
            )

            if response.status_code != 200:
                logging.error(f"OpenAI API error: {response.status_code}, {response.text}")
                return None

            result = response.json()
            chatgpt_output = result["choices"][0]["message"]["content"]
            logging.info(f"API Response: {chatgpt_output}")
            return chatgpt_output

        except requests.exceptions.RequestException as e:
            logging.error(f"Network error: {e}")
            return None

    @staticmethod
    def _parse_json_response(chatgpt_output):
        cleaned = chatgpt_output.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        return json.loads(cleaned)

    def analyse_article_relevance(self, company, title, body):
        """
        Analyzes the title and snippet to determine if the company is the subject of the article.
        Also provides a relevance and usefulness score.

        :param title: Article title.
        :param snippet: Article snippet.
        :param company: Company name to check for relevance.
        :return: Dictionary containing analysis results.
        """
        try:
            logging.info(f"Analyzing article for company: {company}")

            # Construct the prompt for ChatGPT
            prompt = (
                f"Analyze the given article title and body:\n\n"
                f"Title: {title}\nSnippet: {body}\n\n"
                f"### Analysis Criteria:\n"
                f"1. **Is '{company}' the primary subject of the article?** Answer 'Yes' or 'No'.\n"
                f"2. **Is '{company}' mentioned in the article?** Answer 'Yes' or 'No'.\n"
                f"3. **Relevance Score (0-100):** Rate how much the article focuses on {company}.\n"
                f"4. **Usefulness Score (0-100):** Evaluate the investment relevance of the article:\n"
                f"   - Lower scores: Basic price changes, earnings summaries, or general analysis.\n"
                f"   - Higher scores: New projects, debt issuance, dividends, major announcements, controversies, or deep analytical insights.\n\n"
                f"### Expected JSON Response:\n"
                f"Return the response strictly in the following JSON format:\n"
                f'{{"subject": "Yes" or "No", "mentioned": "Yes" or "No", "relevance": <integer 0-100>, "usefulness": <integer 0-100>}}'
            )

            chatgpt_output = self._request_chat_completion(prompt)
            if not chatgpt_output:
                return None

            return self._parse_json_response(chatgpt_output)

        except requests.exceptions.RequestException as e:
            logging.error(f"Network error: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}, Output was: {chatgpt_output}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return None

    def suggest_company_target_queries(self, company_name: str, existing_names: str = "") -> List[Dict[str, str]]:
        """
        Ask GPT to consider all 8 target strategy types and return the best 5 that apply,
        with an explicit strategy_type per query.
        official_exact_quote and official_unquoted are hardcoded by the caller, but GPT
        may still return them — duplicates are filtered in _build_candidates.
        """
        try:
            prompt = (
                "Generate search queries for financial news retrieval for the company below.\n\n"

                f"Primary company name: {company_name}\n"
                f"Existing aliases / identifiers: {existing_names}\n\n"

                "Consider each of the following 8 formulation types, produce one candidate "
                "per type where it applies, then return the best 5 most likely to retrieve "
                "relevant financial news. Skip types that do not apply.\n\n"

                "Formulation types:\n"
                "1. official_exact_quote   — Full official name in double quotes.\n"
                '   e.g. "Pt Adaro Andalan Indonesia Tbk"\n'
                "2. official_unquoted      — Full official name, no quotes.\n"
                "   e.g. Pt Adaro Andalan Indonesia Tbk\n"
                "3. partial_quote_disambiguation — Short name in quotes + brief context to resolve ambiguity.\n"
                '   e.g. "Adaro Andalan" Indonesia coal  |  "Hanwha Life" Korea insurance\n'
                "   Use whenever the short name could match other companies or topics.\n"
                "4. ticker_exchange        — Exchange:Ticker format.\n"
                "   e.g. IDX: AADI  |  NSE: ULTRACEMCO  |  KOSDAQ: 079160\n"
                "   Skip if ticker/exchange is unknown.\n"
                "5. abbreviation_acronym   — Recognised acronym used in financial media.\n"
                "   e.g. ULTRACEMCO  |  TCS  |  PKN\n"
                "   Skip if none exists.\n"
                "6. local_language_name    — Native-script name for non-English companies.\n"
                "   e.g. 아모레퍼시픽  |  中国平安\n"
                "   Skip for English-language companies.\n"
                "7. short_common_name      — Well-known shortened name used in media.\n"
                "   e.g. UltraTech  |  Adaro  |  DiGi\n"
                "   Skip if no well-known short form exists.\n"
                "8. former_name            — Previous official name still appearing in recent news.\n"
                "   e.g. Facebook (for Meta)\n"
                "   Skip if no relevant former name exists.\n\n"

                "Hard rules:\n"
                "- Never append 'stock', 'shares', 'price', 'chart', year numbers, or quarter codes.\n"
                "- Quoted phrases: maximum 4 words inside quotes.\n"
                "- Avoid near-duplicates.\n\n"

                "Return ONLY valid JSON:\n"
                '{"queries": [{"query": "...", "strategy_type": "<type_name>"}, ...]}\n\n'

                "Return exactly 5 queries (fewer only if fewer than 5 types apply)."
            )

            chatgpt_output = self._request_chat_completion(prompt)
            if not chatgpt_output:
                return []

            payload = self._parse_json_response(chatgpt_output)
            queries = payload.get("queries", [])
            if not isinstance(queries, list):
                return []

            results = []
            seen: set = set()
            for item in queries:
                if not isinstance(item, dict):
                    continue
                query = " ".join(str(item.get("query", "")).split()).strip()
                strategy_type = str(item.get("strategy_type", "")).strip()
                if not query or not strategy_type:
                    continue
                key = query.lower()
                if key in seen:
                    continue
                seen.add(key)
                results.append({"query": query, "strategy_type": strategy_type})

            return results[:5]

        except json.JSONDecodeError as e:
            logging.error("JSON parsing error while suggesting company target queries: %s", e)
            return []
        except Exception as e:
            logging.error("Unexpected error suggesting company target queries: %s", e)
            return []

    def suggest_company_search_queries(self, company_name: str, existing_names: str = "", max_queries: int = 3) -> List[Dict[str, str]]:
        """Backwards-compatible wrapper."""
        return self.suggest_company_target_queries(company_name=company_name, existing_names=existing_names)

    def suggest_company_search_names(self, company_name: str, existing_names: str = "", max_names: int = 3):
        """Backwards-compatible wrapper returning only query strings."""
        return [item["query"] for item in self.suggest_company_target_queries(company_name=company_name, existing_names=existing_names)]

    def score_article(self, company: str, title: str, snippet: str) -> Optional[Dict]:
        """
        Score a news article for relevance to a specific company and investment usefulness.

        Uses few-shot examples to anchor scores consistently across different companies
        and article types. Intended for the strategy test harness.

        Returns dict with keys: subject, mentioned, relevance (0-100), usefulness (0-100).
        """
        FEW_SHOT = (
            "Scoring examples - use these to calibrate your scores:\n\n"
            "Example 1 - Company is primary subject, high investment signal:\n"
            "Company: Acme Corp\n"
            'Title: "Acme Corp Announces 500m Share Buyback Programme"\n'
            'Snippet: "Acme Corp said it would repurchase up to 500 million of its own shares over the next '
            '12 months, citing strong cash generation."\n'
            '{"subject": "Yes", "mentioned": "Yes", "relevance": 95, "usefulness": 90}\n\n'
            "Example 2 - Company is primary subject, routine/low signal:\n"
            "Company: Acme Corp\n"
            'Title: "Acme Corp Reports Q3 Earnings in Line With Expectations"\n'
            'Snippet: "Acme Corp posted third-quarter net income of $1.2 billion, matching analyst forecasts. '
            'Revenue rose 3% year-on-year."\n'
            '{"subject": "Yes", "mentioned": "Yes", "relevance": 90, "usefulness": 30}\n\n'
            "Example 3 - Company mentioned but not primary subject:\n"
            "Company: Acme Corp\n"
            'Title: "Global Banks Face Tougher Capital Rules, Analysts Say"\n'
            'Snippet: "Regulators are considering stricter capital requirements. Firms including Acme Corp, '
            'BankX and FinCo could be affected."\n'
            '{"subject": "No", "mentioned": "Yes", "relevance": 20, "usefulness": 10}\n\n'
            "Example 4 - Company not present:\n"
            "Company: Acme Corp\n"
            'Title: "Federal Reserve Signals Pause in Rate Hikes"\n'
            'Snippet: "The Federal Reserve indicated it may hold interest rates steady as inflation eases."\n'
            '{"subject": "No", "mentioned": "No", "relevance": 0, "usefulness": 0}\n\n'
            "Example 5 - Primary subject, high-signal negative event:\n"
            "Company: Acme Corp\n"
            'Title: "Acme Corp Under Investigation for Accounting Irregularities"\n'
            'Snippet: "Regulators have opened a formal investigation into Acme Corp following allegations '
            'of overstated revenues in its 2023 annual report."\n'
            '{"subject": "Yes", "mentioned": "Yes", "relevance": 95, "usefulness": 85}\n'
        )

        prompt = (
            "You are scoring financial news articles for investment relevance.\n\n"
            f"{FEW_SHOT}\n"
            "Now score this article:\n\n"
            f"Company: {company}\n"
            f"Title: {title}\n"
            f"Snippet: {snippet}\n\n"
            "Definitions:\n"
            f"- subject: Is '{company}' the PRIMARY subject? (Yes/No)\n"
            f"- mentioned: Is '{company}' mentioned at all? (Yes/No)\n"
            f"- relevance (0-100): How focused is the article on {company}?\n"
            "  0=not mentioned, 20=mentioned among many, 50=one of several subjects, 90+=primary focus\n"
            "- usefulness (0-100): Investment value of the information:\n"
            "  High (70-100): M&A, fraud/legal issues, dividend changes, buybacks, debt issuance,\n"
            "    major contracts, significant executive changes\n"
            "  Medium (30-69): Guidance updates, analyst rating changes, strategic announcements\n"
            "  Low (0-29): Routine in-line earnings, minor price moves, general sector commentary\n"
            "  Very low (0-10): Auto-generated or templated content with no specific news event —\n"
            "    e.g. titles framed as rhetorical questions about fundamentals ('Is [Company] financially\n"
            "    healthy?', 'Should you buy [Company] today?', 'What does [Company]\\'s debt level mean\n"
            "    for investors?'), or boilerplate data summaries not tied to any announcement.\n\n"
            "Return ONLY valid JSON: "
            '{"subject": "Yes"|"No", "mentioned": "Yes"|"No", "relevance": <0-100>, "usefulness": <0-100>}'
        )

        try:
            chatgpt_output = self._request_chat_completion(prompt, model=self.scoring_model)
            if not chatgpt_output:
                return None
            return self._parse_json_response(chatgpt_output)
        except json.JSONDecodeError as e:
            logging.error("JSON parsing error in score_article: %s", e)
            return None
        except Exception as e:
            logging.error("Unexpected error in score_article: %s", e)
            return None

    def analyse_article_buybacks(self, company_name: str, title: str, snippet: str):
        """
        Analyzes article to determine if it's about the company and mentions buyback/repurchase plans.
        Extracts announced amount and currency if mentioned.

        :param company_name: Full company name to check for relevance.
        :param title: Article title.
        :param snippet: Article snippet/body.
        :return: Dictionary with analysis results in JSON format.
        """
        try:
            logging.info(f"Analyzing buyback article for company: {company_name}")

            # Construct the prompt for ChatGPT
            prompt = (
                f"Analyze the given article title and snippet:\n\n"
                f"Title: {title}\nSnippet: {snippet}\n\n"
                f"### Target Company:\n"
                f"'{company_name}'\n"
                f"(Note: This may contain multiple name variations separated by semicolons - any variation counts)\n\n"
                f"### Analysis Criteria:\n"
                f"1. **Does this article contain information about the target company?**\n"
                f"   - Answer 'Yes' if the target company is mentioned or discussed.\n"
                f"   - Answer 'No' if the article is only about OTHER companies and does NOT mention the target company.\n"
                f"   - CRITICAL: Articles about different companies should be 'No' unless they also mention the target company.\n"
                f"2. **Does the article mention that the target company (specifically) plans to, announces, or approves a buyback or share repurchase?**\n"
                f"   - Answer 'Yes' ONLY if the target company is doing a buyback.\n"
                f"   - Answer 'No' if only other companies are doing buybacks.\n"
                f"   - Focus on NEW announcements or plans, not historical buybacks already completed.\n"
                f"3. **If the target company has a buyback amount mentioned, extract:**\n"
                f"   - The announced amount as a full number (e.g., '$25 billion' = 25000000000, 'EUR500m' = 500000000)\n"
                f"   - The currency in ISO 4217 format (e.g., 'USD', 'EUR', 'GBP', 'JPY')\n"
                f"   - If no specific amount OR buyback is for a different company, set both to null\n\n"
                f"### Expected JSON Response:\n"
                f"Return the response ONLY as valid JSON in this exact format:\n"
                f'{{"Relevant": "Yes"|"No", "Buyback": "Yes"|"No", "Amount": <number>|null, "Currency": "<ISO_CODE>"|null}}\n\n'
                f"### Examples:\n"
                f"Example 1 - Target company with buyback:\n"
                f'Target: "Company A"\n'
                f'Title: "Company A Announces $25 Billion Share Buyback Program"\n'
                f'Response: {{"Relevant": "Yes", "Buyback": "Yes", "Amount": 25000000000, "Currency": "USD"}}\n\n'
                f"Example 2 - Different company doing buyback:\n"
                f'Target: "Company A"\n'
                f'Title: "Company B Buying Back Shares Daily"\n'
                f'Response: {{"Relevant": "No", "Buyback": "No", "Amount": null, "Currency": null}}\n\n'
                f"Example 3 - Multiple companies, only others doing buyback:\n"
                f'Target: "Company A"\n'
                f'Title: "Company B and Company C Announce Buybacks, Company A Reports Revenue"\n'
                f'Response: {{"Relevant": "Yes", "Buyback": "No", "Amount": null, "Currency": null}}'
            )

            chatgpt_output = self._request_chat_completion(prompt)
            if not chatgpt_output:
                return None

            return self._parse_json_response(chatgpt_output)

        except requests.exceptions.RequestException as e:
            logging.error(f"Network error: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}, Output was: {chatgpt_output}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return None
