# tests/test_export_xlsx.py
import csv
from pathlib import Path
from openpyxl import load_workbook
from src import export_xlsx

def test_csv_to_xlsx_identical_content(tmp_path):
    csv_path = tmp_path / "s.csv"
    rows = [["candidate_id", "rank", "score", "reasoning"],
            ["CAND_0000001", "1", "0.9876", "good fit"],
            ["CAND_0000002", "2", "0.9001", "ok fit"]]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    xlsx_path = tmp_path / "s.xlsx"
    export_xlsx.csv_to_xlsx(csv_path, xlsx_path)
    wb = load_workbook(xlsx_path)
    ws = wb.active
    got = [[str(c.value) for c in row] for row in ws.iter_rows()]
    assert got[0] == ["candidate_id", "rank", "score", "reasoning"]
    assert got[1][0] == "CAND_0000001"
    assert len(got) == 3
