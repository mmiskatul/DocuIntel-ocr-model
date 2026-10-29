"""PyMuPDF digital PDF parser with DPI-aware raster OCR fallback."""
from __future__ import annotations

from typing import List, Tuple
from PIL import Image
import pymupdf

from src.schemas.ocr import OCRBoxSchema
from src.services.ocr_engine import ocr_service


class PDFParserService:
    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def parse_pdf(self, content: bytes, max_pages: int = 10) -> Tuple[str, float, List[OCRBoxSchema], int]:
        """
        Parses PDF documents using PyMuPDF:
        1. Extracts digital text streams directly from native PDFs.
        2. Renders pages at 150 DPI into pixmaps for image-level OCR bounding boxes.
        3. Fuses digital text with coordinate bounding boxes.
        """
        doc = pymupdf.open(stream=content, filetype="pdf")
        page_count = len(doc)
        all_texts: List[str] = []
        all_boxes: List[OCRBoxSchema] = []
        total_conf = 0.0
        pages_processed = 0

        for page_idx in range(min(page_count, max_pages)):
            page = doc[page_idx]
            digital_text = page.get_text().strip()

            # Render page to raster image for OCR bounding boxes
            pix = page.get_pixmap(dpi=self.dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text, ocr_conf, boxes = ocr_service.run_ocr(img)

            # Prioritize digital text stream if present
            page_text = digital_text if len(digital_text) > len(ocr_text) * 0.7 else ocr_text
            if page_text:
                all_texts.append(f"--- Page {page_idx + 1} ---\n{page_text}")
            all_boxes.extend(boxes)
            total_conf += ocr_conf if ocr_conf > 0 else 98.0
            pages_processed += 1

        combined_text = "\n\n".join(all_texts)
        mean_conf = round(total_conf / max(1, pages_processed), 2)
        return combined_text, mean_conf, all_boxes, page_count


pdf_parser_service = PDFParserService()

