"""Retrieve a short, dated excerpt from a supported issuer product page."""

import asyncio
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Literal
from urllib.parse import urlsplit

LiveTopic = Literal["annual_fee", "grocery", "gas", "online_retail", "transit", "streaming"]
TOPICS = {
    "annual_fee": r"annual fee",
    "grocery": r"(?:U\.?S\.?\s+)?supermarkets?",
    "gas": r"(?:U\.?S\.?\s+)?gas stations?",
    "online_retail": r"(?:U\.?S\.?\s+)?online retail",
    "transit": r"\btransit\b",
    "streaming": r"streaming subscriptions?",
}
MAX_HTML_BYTES = 2_000_000


class _ProductHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.headings = []
        self.heading = None
        self.hidden = None

    def handle_starttag(self, tag, attrs):
        if self.hidden:
            return
        if tag in {"script", "style", "noscript", "svg", "template"}:
            self.hidden = tag
        elif tag in {"title", "h1"}:
            self.heading = tag
            self.headings.append("")
        elif tag in {"p", "div", "br", "li", "h2", "h3", "h4", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == self.hidden:
            self.hidden = None
        elif self.hidden is None and tag == self.heading:
            self.heading = None

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)
            if self.heading:
                self.headings[-1] += data


def product_excerpt(html: str, product_name: str, topic: LiveTopic) -> str | None:
    """Extract a short topic excerpt only from a page identifying the expected card."""
    page = _ProductHTML()
    page.feed(html)
    product_pattern = rf"\b{re.escape(product_name)}\b"
    if not any(re.search(product_pattern, heading, re.I) for heading in page.headings):
        return None
    for paragraph in "".join(page.parts).splitlines():
        paragraph = " ".join(paragraph.split())
        for match in re.finditer(TOPICS[topic], paragraph, re.I):
            before = paragraph[:match.start()].split()[-7:]
            after = paragraph[match.end():].split()[:11]
            excerpt = " ".join([*before, match.group(), *after])
            has_value = (
                re.search(r"\$\s*\d|\bno annual fee\b", excerpt, re.I)
                if topic == "annual_fee"
                else re.search(r"\d\s*%", excerpt)
            )
            if has_value:
                return excerpt
    return None


async def fetch_product_excerpt(url: str, product_name: str, topic: LiveTopic) -> str | None:
    """Fetch an issuer page without following redirects or loading scripts."""
    import httpx

    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "www.americanexpress.com":
        raise ValueError("Unsupported issuer URL.")

    async def read_page():
        async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
            async with client.stream("GET", url, headers={"Accept": "text/html"}) as response:
                response.raise_for_status()
                if "text/html" not in response.headers.get("content-type", "").lower():
                    return None
                content = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65_536):
                    if len(content) + len(chunk) > MAX_HTML_BYTES:
                        return None
                    content.extend(chunk)
                return product_excerpt(content.decode("utf-8", errors="replace"), product_name, topic)

    return await asyncio.wait_for(read_page(), timeout=9)


async def check_product_page(card_id, product_name, url, topic: LiveTopic) -> dict:
    """Return source text for manual reconciliation, or report an explicit retrieval gap."""
    import httpx

    try:
        excerpt = await fetch_product_excerpt(url, product_name, topic)
    except (asyncio.TimeoutError, httpx.HTTPError):
        excerpt = None
    if excerpt is None:
        return {
            "card_id": card_id,
            "evidence": [],
            "gaps": [f"Could not verify {topic} from the live issuer product page. Use dated evidence if available."],
        }
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "card_id": card_id,
        "evidence": [{
            "card_id": card_id,
            "field": f"{topic}_live_excerpt",
            "value": excerpt,
            "source_url": url,
            "source_date": checked_at[:10],
            "retrieved_at": checked_at,
            "status": "issuer_live_excerpt",
        }],
        "gaps": ["The excerpt requires interpretation and may omit details present in the full terms."],
    }
