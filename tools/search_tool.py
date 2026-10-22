"""Wikipedia retrieval with explicit timeout and typed failures."""
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from tools.types import ToolResult


class WikipediaSearch:
    def __init__(self, timeout=15, min_interval=1.0, max_results=2, snippet_chars=1000):
        self.timeout, self.min_interval = timeout, min_interval
        self.max_results, self.snippet_chars = max_results, snippet_chars
        self.last_request = 0.0

    def _request(self, parameters):
        time.sleep(max(0.0, self.min_interval - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        url = "https://en.wikipedia.org/w/api.php?" + urlencode({"format": "json", **parameters})
        request = Request(url, headers={"User-Agent": "AgentEvalProject/2.0 (educational retrieval)"})
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def __call__(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 300:
            return ToolResult(False, "Use a short, nonempty entity query.", "invalid_query")
        try:
            result = self._request({"action": "query", "list": "search", "srsearch": query,
                                    "srlimit": self.max_results})
            hits = result.get("query", {}).get("search", [])
            if not hits:
                return ToolResult(False, "No matching page. Try a different entity.", "no_results")
            pages = self._request({"action": "query", "prop": "extracts|pageprops", "explaintext": 1,
                                   "exintro": 1, "pageids": "|".join(str(x["pageid"]) for x in hits)})
            output = []
            for page in pages.get("query", {}).get("pages", {}).values():
                title, extract = page.get("title", ""), page.get("extract", "")
                if "disambiguation" in page.get("pageprops", {}):
                    extract = "Ambiguous entity; refine the query. " + extract
                output.append(f"Title: {title}\nEvidence: {extract[:self.snippet_chars]}")
            if not output:
                return ToolResult(False, "No readable page content.", "empty_page")
            return ToolResult(True, "\n\n".join(output))
        except HTTPError as exc:
            return ToolResult(False, f"Wikipedia HTTP {exc.code}.", "rate_limit" if exc.code == 429 else "http_error")
        except (URLError, TimeoutError):
            return ToolResult(False, "Wikipedia request failed or timed out.", "network_error")
        except (ValueError, KeyError, TypeError):
            return ToolResult(False, "Invalid Wikipedia response.", "response_error")


def search(query, max_results=2):
    return WikipediaSearch(max_results=max_results)(query).text
