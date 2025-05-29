import csv
from pathlib import Path


def get_symbols_from_csv(
    csv_path="configs/nasdaq_screener.csv",
    country=None,
    sector=None,
    limit=None,
):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"{csv_path} not found!")
    symbols = set()
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if country and row.get("country") != country:
                continue
            if sector and row.get("sector") != sector:
                continue
            symbol = row.get("symbol", "").strip()
            if symbol:
                symbols.add(symbol)
                if limit and len(symbols) >= limit:
                    break
    return sorted(symbols)
