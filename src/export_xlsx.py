"""Convert the validator-clean CSV into an identical XLSX for portal upload."""
import argparse
import csv
from pathlib import Path
from openpyxl import Workbook

def csv_to_xlsx(csv_path: Path, xlsx_path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "ranking"
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            ws.append(row)
    wb.save(xlsx_path)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    csv_to_xlsx(Path(args.csv), Path(args.out))

if __name__ == "__main__":
    main()
