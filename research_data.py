"""Full-catalog discovery and conservative matching to the complete CFPB survey."""

import csv
import json
import re
import unicodedata
from decimal import Decimal
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).with_name("data")
CATALOG_FILE = DATA_DIR / "cards_67credit.json"
CFPB_FILE = DATA_DIR / "cfpb_tccp_2025-12-31.csv"
ISSUER_ALIASES = {
    "american express": {"american express national bank"},
    "bank of america": {"bank of america national association"},
    "barclays": {"barclays bank delaware"},
    "capital one": {"capital one national association"},
    "chase": {"jpmorgan chase bank national association"},
    "citi": {"citibank n a"},
    "citizens": {"citizens bank national association"},
    "discover": {"discover bank", "capital one national association"},
    "apple": {"goldman sachs bank usa"},
    "navy federal": {"navy federal credit union"},
    "penfed": {"pentagon federal credit union"},
    "synchrony": {"synchrony financial", "synchrony bank"},
    "us bank": {"u s bancorp", "u s bank national association"},
    "usaa": {"usaa federal savings bank"},
    "wells fargo": {"wells fargo bank national association"},
}
CATEGORY_KEYS = {
    "grocery": "groceries", "online_retail": "online", "transit": "other",
    # The catalog has no typed entertainment field. Use its conservative base-rate
    # bucket for discovery only; issuer verification can register the actual rate.
    "entertainment": "other",
}
SPEND_CATEGORIES = {
    "grocery", "dining", "entertainment", "travel", "gas", "online_retail",
    "transit", "streaming", "other",
}


def words(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


@lru_cache(maxsize=1)
def catalog() -> dict:
    payload = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    return {card["slug"]: card for card in payload["cards"]}


@lru_cache(maxsize=4)
def survey(path: Path = CFPB_FILE) -> tuple[dict, ...]:
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return tuple(csv.DictReader(file))


def search_cards(query: str = "", issuer: str = "", max_annual_fee_usd: float | None = None,
                 include_business: bool = False, limit: int = 10) -> dict:
    """Find active cards across the complete catalog, preserving missing values."""
    if not 1 <= limit <= 30:
        raise ValueError("Limit must be between 1 and 30.")
    if max_annual_fee_usd is not None and (not 0 <= max_annual_fee_usd <= 1_000_000):
        raise ValueError("Annual fee limit must be finite and nonnegative.")
    tokens = words(query).split()
    results = []
    for card in catalog().values():
        searchable = words(card["name"] + " " + card["issuer"])
        if card["status"] != "active" or ((card["category"] == "business" or "business" in words(card["name"]).split()) and not include_business):
            continue
        if issuer and words(card["issuer"]) != words(issuer):
            continue
        if not all(token in searchable for token in tokens):
            continue
        fee = card["annualFee"]
        if max_annual_fee_usd is not None and (fee is None or fee > max_annual_fee_usd):
            continue
        results.append(dict(card))
    return {"catalog_count": len(catalog()), "matched_count": len(results), "cards": results[:limit]}


def shortlist_cards(monthly_spend: dict, max_annual_fee_usd: float | None = None,
                    include_business: bool = False, limit: int = 5) -> dict:
    """Screen candidates using catalog rates at their stated point value before issuer verification."""
    if not 1 <= limit <= 20:
        raise ValueError("Limit must be between 1 and 20.")
    if monthly_spend.keys() - SPEND_CATEGORIES:
        raise ValueError("Unknown spending category.")
    spending = {category: Decimal(str(value)) for category, value in monthly_spend.items()}
    if not spending or any(not value.is_finite() or not 0 <= value <= Decimal('1e12') for value in spending.values()):
        raise ValueError("Supply finite, nonnegative monthly spending amounts.")
    if max_annual_fee_usd is not None and not 0 <= max_annual_fee_usd <= 1_000_000:
        raise ValueError("Annual fee limit must be finite and nonnegative.")
    results = []
    missing = []
    for card in catalog().values():
        if card["status"] != "active" or ((card["category"] == "business" or "business" in words(card["name"]).split()) and not include_business):
            continue
        fee = card["annualFee"]
        if max_annual_fee_usd is not None and (fee is None or fee > max_annual_fee_usd):
            continue
        rates = card["rewardsRates"]
        required = [CATEGORY_KEYS.get(category, category) for category in spending]
        if fee is None or rates is None or any(rates.get(key) is None for key in [*required, "pointValue"]):
            missing.append(card["slug"])
            continue
        gross = sum(spend * 12 * Decimal(str(rates[key])) * Decimal(str(rates["pointValue"]))
                    for (category, spend), key in zip(spending.items(), required))
        results.append({
            "card_id": card["slug"], "name": card["name"], "issuer": card["issuer"],
            "annual_fee_usd": fee,
            "catalog_screening_value_usd": float((gross - Decimal(str(fee))).quantize(Decimal('.01'))),
            "rewards_notes": card["rewardsNotes"], "point_value_usd": rates["pointValue"],
            "source_url": card["url"], "source_date": card["lastVerified"],
            "status": "requires_issuer_verification",
        })
    results.sort(key=lambda item: (-item["catalog_screening_value_usd"], item["card_id"]))
    return {
        "catalog_count": len(catalog()), "scored_count": len(results), "candidates": results[:limit],
        "unscored_card_ids": missing,
        "assumptions": [
            "Catalog screening uses the dataset's baseline point value and ignores caps, activation, credits and bonuses.",
            "Review rewards notes and official terms before ranking final net values. Transit uses the catalog's other rate.",
        ],
    }


def product_key(name: str, issuer: str) -> str:
    normalized = words(name)
    for prefix in (words(issuer), "american express", "jpmorgan chase", "chase", "citi"):
        normalized = re.sub(rf"\b{re.escape(prefix)}\b", " ", normalized)
    normalized = re.sub(r"\b(?:credit|card|from|the|by|visa|mastercard|signature|sm|r)\b", " ", normalized)
    return " ".join(normalized.split())


def match_cfpb(card: dict, path: Path = CFPB_FILE) -> dict:
    issuer = words(card["issuer"])
    allowed = ISSUER_ALIASES.get(issuer, {issuer})
    issuer_rows = [row for row in survey(path) if words(row.get("Institution Name", "")) in allowed]
    key = product_key(card["name"], card["issuer"])
    exact = [row for row in issuer_rows if product_key(row.get("Product Name", ""), card["issuer"]) == key]
    if len(exact) == 1:
        return {"status": "matched", "method": "issuer_alias_and_normalized_product", "row": dict(exact[0])}
    if len(exact) > 1:
        return {"status": "ambiguous", "candidate_count": len(exact), "candidates": [_summary(row) for row in exact[:5]]}
    suggested = sorted(issuer_rows, key=lambda row: SequenceMatcher(
        None, key, product_key(row.get("Product Name", ""), card["issuer"])).ratio(), reverse=True)[:3]
    return {"status": "not_found", "candidates": [_summary(row) for row in suggested]}


def _summary(row: dict) -> dict:
    fields = ("Workbook Row", "Institution Name", "Product Name", "Report Date", "Annual Fee",
              "Purchase APR min", "Purchase APR median", "Purchase APR max")
    return {name: row.get(name, "") for name in fields}


def search_cfpb_products(query: str, issuer: str = "", limit: int = 10) -> dict:
    """Search the complete survey independently of catalog coverage or matching."""
    if not 1 <= limit <= 30:
        raise ValueError("Limit must be between 1 and 30.")
    tokens = words(query).split()
    canonical = words(issuer)
    allowed = ISSUER_ALIASES.get(canonical, {canonical})
    results = []
    for row in survey():
        if issuer and words(row["Institution Name"]) not in allowed:
            continue
        if all(token in words(row["Product Name"]) for token in tokens):
            results.append(_summary(row))
    return {"survey_count": len(survey()), "matched_count": len(results), "records": results[:limit]}
