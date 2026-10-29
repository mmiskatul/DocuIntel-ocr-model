"""Document Parsing & OCR Engine (PyMuPDF + RapidOCR / PaddleOCR ONNX)

Provides:
- RapidOCR ONNX runtime inference with reading-order sorting and bounding box extraction
- PyMuPDF native digital PDF parsing with raster fallback
- DPI-aware rendering for scanned invoice text extraction
"""
from __future__ import annotations

import io
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageOps
import pymupdf
from rapidocr_onnxruntime import RapidOCR

from schemas import OCRBoxSchema


class OCRParserEngine:
    def __init__(self):
        self._ocr = RapidOCR()

    def run_ocr(self, img: Image.Image) -> Tuple[str, float, List[OCRBoxSchema]]:
        """Executes RapidOCR with natural reading-order sorting."""
        arr = np.asarray(ImageOps.exif_transpose(img).convert("RGB"))
        result, _ = self._ocr(arr)
        if not result:
            return "", 0.0, []

        # Sort boxes into top-to-bottom, left-to-right reading order
        boxes = sorted(((sum(p[1] for p in b) / 4, min(p[0] for p in b), t, float(c)) for b, t, c in result))
        heights = [max(p[1] for p in b) - min(p[1] for p in b) for b, _, _ in result]
        tol = max(8.0, sorted(heights)[len(heights) // 2] * 0.6)
        lines: list[list[tuple[float, str]]] = []
        last_y = None
        for y, x, t, _ in boxes:
            if last_y is None or abs(y - last_y) > tol:
                lines.append([])
            lines[-1].append((x, t))
            last_y = y

        ordered_text = "\n".join("  ".join(t for _, t in sorted(line)) for line in lines)
        mean_conf = sum(float(c) for *_, c in result) / len(result) * 100.0

        raw_boxes = [
            OCRBoxSchema(box=[list(pt) for pt in b], text=t, confidence=float(c))
            for b, t, c in result
        ]
        return ordered_text, round(mean_conf, 2), raw_boxes

    def parse_pdf(self, content: bytes, max_pages: int = 10) -> Tuple[str, float, List[OCRBoxSchema], int]:
        """
        Parses digital and scanned PDF documents using PyMuPDF:
        1. Extracts high-fidelity digital text streams from native digital PDFs.
        2. Renders pages at 150 DPI into image pixmaps.
        3. Runs RapidOCR for bounding boxes and fallback for scanned receipts.
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
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text, ocr_conf, boxes = self.run_ocr(img)

            # Digital text is prioritized if available; OCR text used for scans
            page_text = digital_text if len(digital_text) > len(ocr_text) * 0.7 else ocr_text
            if page_text:
                all_texts.append(f"--- Page {page_idx + 1} ---\n{page_text}")
            all_boxes.extend(boxes)
            total_conf += ocr_conf if ocr_conf > 0 else 98.0
            pages_processed += 1

        combined_text = "\n\n".join(all_texts)
        mean_conf = round(total_conf / max(1, pages_processed), 2)
        return combined_text, mean_conf, all_boxes, page_count


ocr_engine = OCRParserEngine()
