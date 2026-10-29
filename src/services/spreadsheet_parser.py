"""Spreadsheet parsing service using pandas and openpyxl."""
from __future__ import annotations

import io
from typing import Any, Dict, List
import openpyxl
import pandas as pd

from src.schemas.spreadsheet import SpreadsheetRowSchema


class SpreadsheetParserService:
    def parse(self, content: bytes, filename: str) -> dict:
        """Parses Excel (.xlsx/.xls) and CSV spreadsheets, extracting financial rows."""
        buf = io.BytesIO(content)
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(buf, engine="openpyxl")
        else:
            df = pd.read_csv(buf)

        df.columns = [str(c).strip().lower() for c in df.columns]
        rows: List[SpreadsheetRowSchema] = []
        total_amt = 0.0

        for idx, row in df.iterrows():
            row_dict = row.dropna().to_dict()
            amt = None
            for col_name in ["amount", "total", "net", "price", "debit", "credit"]:
                if col_name in row_dict:
                    try:
                        amt = float(str(row_dict[col_name]).replace("$", "").replace("€", "").replace("৳", "").replace(",", "").strip())
                        break
                    except ValueError:
                        pass

            txn = str(row_dict.get("transaction") or row_dict.get("description") or row_dict.get("vendor") or f"Item {idx + 1}")
            dt = str(row_dict.get("date") or row_dict.get("invoice_date") or "")
            inv = str(row_dict.get("invoice_number") or row_dict.get("invoice") or row_dict.get("ref") or "")
            supp = str(row_dict.get("supplier") or row_dict.get("vendor") or "")

            if amt is not None:
                total_amt += amt

            rows.append(SpreadsheetRowSchema(
                row_index=idx + 1,
                transaction=txn,
                date=dt or None,
                invoice_number=inv or None,
                amount=amt,
                supplier=supp or None,
                raw_data={str(k): str(v) for k, v in row_dict.items()},
            ))

        return {
            "filename": filename,
            "row_count": len(rows),
            "columns": [str(c) for c in df.columns],
            "total_amount": round(total_amt, 2),
            "rows": rows,
        }


spreadsheet_parser_service = SpreadsheetParserService()

