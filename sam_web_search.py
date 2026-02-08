from __future__ import annotations

import html
import re
import time
from typing import Any, Dict, List, Optional

import requests

SAM_WEB_SEARCH_AVAILABLE = True

_search_context: Dict[str, Any] = {}


def initialize_sam_web_search(google_drive: Optional[Any] = None) -> None:
    _search_context["google_drive"] = google_drive


def _parse_duckduckgo_html(text: str, max_results: int) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    pattern = re.compile(r"<a rel=\"nofollow\" class=\"result__a\" href=\"(.*?)\".*?>(.*?)</a>", re.S)
    for match in pattern.finditer(text):
        url = html.unescape(match.group(1))
        title = re.sub("<.*?>", "", match.group(2))
        results.append({"title": title, "url": url, "snippet": ""})
        if len(results) >= max_results:
            return results

    # Lite HTML fallback
    lite_pattern = re.compile(r"<a rel=\"nofollow\" class=\"result-link\" href=\"(.*?)\".*?>(.*?)</a>", re.S)
    for match in lite_pattern.finditer(text):
        url = html.unescape(match.group(1))
        title = re.sub("<.*?>", "", match.group(2))
        results.append({"title": title, "url": url, "snippet": ""})
        if len(results) >= max_results:
            break
    return results


def search_web_with_sam(query: str, save_to_drive: bool = False, max_results: int = 5) -> Dict[str, Any]:
    if not query.strip():
        return {"query": query, "results": [], "source": "duckduckgo", "timestamp": time.time()}

    headers = {
        "User-Agent": "SAM/2.0 (web search)"
    }
    params = {"q": query}
    response = requests.get("https://duckduckgo.com/html/", params=params, headers=headers, timeout=20)
    response.raise_for_status()

    results = _parse_duckduckgo_html(response.text, max_results)
    payload = {
        "query": query,
        "results": results,
        "source": "duckduckgo",
        "timestamp": time.time(),
    }

    if save_to_drive:
        drive = _search_context.get("google_drive")
        if drive and hasattr(drive, "save_web_search"):
            try:
                drive.save_web_search(payload)
            except Exception:
                pass

    return payload
