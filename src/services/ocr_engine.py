"""RapidOCR (PaddleOCR ONNX Engine) service for deep learning text detection and recognition."""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR

from src.schemas.ocr import OCRBoxSchema


class OCREngineService:
    def __init__(self):
        self._ocr = RapidOCR()

    def run_ocr(self, img: Image.Image) -> Tuple[str, float, List[OCRBoxSchema]]:
        """Performs deep-learning OCR with natural reading-order sorting and bounding box extraction."""
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


ocr_service = OCREngineService()

