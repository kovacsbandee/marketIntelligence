import csv
from pathlib import Path

def get_symbols_from_csv(
    csv_path="configs/nasdaq_screener.csv",
    country=None,
    sector=None,
    limit=None,
):
    """
    Extracts stock symbols from a CSV file with optional filtering and limiting.

    Args:
        csv_path (str): Path to the CSV file containing stock data.
        country (str, optional): If provided, only include symbols with this country value.
        sector (str, optional): If provided, only include symbols with this sector value.
        limit (int, optional): If provided, return at most this many symbols.

    Returns:
        list[str]: Sorted list of unique stock symbols matching the filters.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.

    Notes:
        - The CSV file must have at least 'symbol', 'country', and 'sector' columns.
        - Duplicate symbols are removed.
        - Filtering is applied before limiting.
    """
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
