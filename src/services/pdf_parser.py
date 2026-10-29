"""PyMuPDF digital PDF parser using native high-speed C++ engine (100% model-free)."""
from __future__ import annotations

from typing import List, Tuple
import pymupdf

from src.schemas.ocr import OCRBoxSchema


class PDFParserService:
    def parse_pdf(self, content: bytes, max_pages: int = 10) -> Tuple[str, float, List[OCRBoxSchema], int]:
        """
        Parses PDF documents using PyMuPDF:
        1. Extracts digital text streams directly from native PDFs with zero AI model overhead.
        2. Extracts word bounding boxes directly from vector coordinate streams.
        """
        doc = pymupdf.open(stream=content, filetype="pdf")
        page_count = len(doc)
        all_texts: List[str] = []
        all_boxes: List[OCRBoxSchema] = []
        pages_processed = 0

        for page_idx in range(min(page_count, max_pages)):
            page = doc[page_idx]
            digital_text = page.get_text().strip()
            if digital_text:
                all_texts.append(f"--- Page {page_idx + 1} ---\n{digital_text}")

            # Extract word bounding boxes natively from PDF vector stream
            words = page.get_text("words")
            for w in words:
                # w is (x0, y0, x1, y1, "word", block_no, line_no, word_no)
                box = [[float(w[0]), float(w[1])], [float(w[2]), float(w[1])], [float(w[2]), float(w[3])], [float(w[0]), float(w[3])]]
                all_boxes.append(OCRBoxSchema(box=box, text=w[4], confidence=99.0))
            pages_processed += 1

        combined_text = "\n\n".join(all_texts)
        mean_conf = 99.0 if all_texts else 0.0
        return combined_text, mean_conf, all_boxes, page_count


pdf_parser_service = PDFParserService()
