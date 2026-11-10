"""Download complete public datasets and preserve source metadata for CardWise."""

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
CATALOG_URL = "https://67.credit/api/v1/cards.json"
CFPB_URL = "https://files.consumerfinance.gov/f/documents/cfpb_tccp-data_2025-12-31.xlsx"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def convert_workbook(path: Path, output: Path) -> int:
    """Export every product and original column, converting percentage units explicitly."""
    with ZipFile(path) as workbook:
        strings = [
            "".join(node.text or "" for node in item.findall(".//s:t", NS))
            for item in ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        ]
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    rows = []
    header = None
    for row in sheet.findall(".//s:row", NS):
        values = {}
        for cell in row.findall("s:c", NS):
            column = re.sub(r"\d", "", cell.get("r"))
            value = cell.findtext("s:v", namespaces=NS)
            if cell.get("t") == "s" and value is not None:
                value = strings[int(value)]
            elif cell.get("t") == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall(".//s:t", NS))
            values[column] = value or ""
        if {"Institution Name", "Product Name", "Report Date"}.issubset(values.values()):
            header = {column: name for column, name in values.items() if name}
            continue
        if header is None:
            continue
        record = {name: "\n".join(line.rstrip() for line in values.get(column, "").splitlines())
                  for column, name in header.items()}
        if not record.get("Institution Name") or not record.get("Product Name") or not record.get("Report Date"):
            continue
        for name, value in record.items():
            if name.startswith(("Purchase APR ", "Intro APR ", "Transfer APR ", "Advance APR ")) or "(%)" in name:
                try:
                    record[name] = str((Decimal(value) * 100).quantize(Decimal("0.0001")).normalize())
                except InvalidOperation:
                    pass
        rows.append({"Workbook Row": row.get("r"), **record})
    if not rows:
        raise ValueError("No CFPB product records found in the workbook.")
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Workbook Row", *header.values()], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-file", type=Path)
    parser.add_argument("--cfpb-workbook", type=Path)
    args = parser.parse_args()
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    sources = []
    for provided, url, filename in (
        (args.catalog_file, CATALOG_URL, "cards_67credit.json"),
        (args.cfpb_workbook, CFPB_URL, "cfpb_tccp_2025-12-31.xlsx"),
    ):
        if provided:
            content = provided.read_bytes()
        else:
            with urlopen(url, timeout=60) as response:
                content = response.read()
        target = directory / filename
        target.write_bytes(content)
        sources.append({"file": filename, "url": url, "sha256": hashlib.sha256(content).hexdigest()})
    catalog = json.loads((directory / "cards_67credit.json").read_text(encoding="utf-8"))
    cards = catalog["cards"]
    if len({card["slug"] for card in cards}) != len(cards):
        raise ValueError("Duplicate catalog slugs.")
    count = convert_workbook(directory / "cfpb_tccp_2025-12-31.xlsx", directory / "cfpb_tccp_2025-12-31.csv")
    metadata = {
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "catalog_count": len(cards), "cfpb_count": count,
        "catalog_metadata": catalog["meta"], "sources": sources,
        "cfpb_percentage_units": "Percent, converted from workbook fractional rates.",
    }
    (directory / "research_sources.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"catalog_count": len(cards), "cfpb_count": count}))


if __name__ == "__main__":
    main()
