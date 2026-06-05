import logging
import os
import re
import requests
from dotenv import load_dotenv
import json
from typing import Dict, List

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
        re.search(r"\b[A-Z0-9]{1,6}[.:][A-Z]{1,5}\b", query)
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

    def __init__(self, model="gpt-4o-mini"):
        """
        Initializes the ChatGPTAnalyser class with API credentials and optional proxy settings.

        :param api_key: OpenAI API key.
        :param proxy: Proxy URL (e.g., "http://your.proxy.server:8080") or None.
        :param model: The OpenAI model to use.
        """
        self.api_key = OPENAI_API_KEY
        self.model = model

        # Configure proxy settings
        self.proxies = {
            key: value
            for key, value in {"http": HTTP_PROXY, "https": HTTPS_PROXY}.items()
            if value
        }

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("ChatGPTAnalyser initialized.")

    def _request_chat_completion(self, prompt):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
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

    def suggest_company_target_queries(self, company_name: str, existing_names: str = "", max_names: int = 3) -> List[Dict[str, str]]:
        """
        Suggest distinct company-targeting formulations for financial-news retrieval.

        :param company_name: Primary entity name.
        :param existing_names: Existing semicolon-delimited aliases from the entity table.
        :param max_names: Maximum number of formulations to return.
        :return: List of dictionaries containing query and strategy_type.
        """
        try:
            prompt = (
                "Generate company-targeting search queries for financial news retrieval.\n\n"

                f"Primary company name: {company_name}\n"
                f"Existing aliases / identifiers: {existing_names}\n\n"

                "Generate distinct search formulations likely to appear in financial news headlines, "
                "snippets, filings, and press releases.\n\n"

                "Possible formulation types:\n"
                "- official company name in quotes\n"
                "- official company name without quotes\n"
                "- short/common company name\n"
                "- company name with ticker or exchange\n"
                "- abbreviation/acronym if commonly used\n"
                "- partially quoted company name with contextual disambiguation\n"
                "- local-language company name if relevant\n\n"

                "Guidelines:\n"
                "- Use quotes only when they improve precision.\n"
                "- For ambiguous names, prefer partial quoting with context.\n"
                "- Example: '\"Continental\" german tyres'\n"
                "- Avoid duplicate or near-duplicate queries.\n"
                "- Avoid excessively restrictive long quoted phrases.\n"
                "- Avoid overly generic one-word queries unless distinctive.\n"
                "- Prefer formulations likely to appear naturally in financial news.\n\n"

                "Return ONLY valid JSON:\n"
                "{\n"
                '  "queries": [\n'
                '    "\\"Continental\\" german tyres"\n'
                "  ]\n"
                "}\n\n"

                f"Return no more than {max_names} queries."
            )

            chatgpt_output = self._request_chat_completion(prompt)
            if not chatgpt_output:
                return []

            payload = self._parse_json_response(chatgpt_output)
            queries = payload.get("queries", [])
            if not isinstance(queries, list):
                return []

            clean_queries = []
            seen = set()
            for item in queries:
                query = item.get("query") if isinstance(item, dict) else item
                if not isinstance(query, str):
                    continue
                cleaned_query = " ".join(query.split())
                if not cleaned_query:
                    continue
                key = cleaned_query.lower()
                if key in seen:
                    continue
                seen.add(key)

                clean_queries.append(
                    {
                        "query": cleaned_query,
                        "strategy_type": _infer_target_strategy_type(
                            cleaned_query,
                            company_name=company_name,
                            existing_names=existing_names,
                        ),
                    }
                )

            return clean_queries[:max_names]

        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error while suggesting company target queries: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error suggesting company target queries: {e}")
            return []

    def suggest_company_search_queries(self, company_name: str, existing_names: str = "", max_queries: int = 3) -> List[Dict[str, str]]:
        """
        Backwards-compatible wrapper for the company-targeting generator.
        """
        return self.suggest_company_target_queries(
            company_name=company_name,
            existing_names=existing_names,
            max_names=max_queries,
        )

    def suggest_company_search_names(self, company_name: str, existing_names: str = "", max_names: int = 3):
        """
        Backwards-compatible wrapper returning only query strings.
        """
        query_records = self.suggest_company_search_queries(
            company_name=company_name,
            existing_names=existing_names,
            max_queries=max_names,
        )
        return [item["query"] for item in query_records]

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
                f"   - The announced amount as a full number (e.g., '$25 billion' = 25000000000, '€500m' = 500000000)\n"
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
