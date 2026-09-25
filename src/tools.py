from langchain_tavily import TavilySearch
from typing import List

from dotenv import load_dotenv
load_dotenv()


def tavily_search_tool(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearch(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized_results: List[dict] = []

    for r in results.get("results", []) or []:
        normalized_results.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "published_at": r.get("published_at") or r.get("published_date"),
            "snippet": r.get("snippet") or r.get("content") or "",
            "source": r.get("source")
        })

    return normalized_results
