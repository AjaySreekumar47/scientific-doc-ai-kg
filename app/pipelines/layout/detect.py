from __future__ import annotations

from io import BytesIO
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import fitz  # PyMuPDF

import layoutparser as lp
from layoutparser.models.paddledetection import PaddleDetectionLayoutModel


# PubLayNet label mapping (common convention)
LABEL_MAP = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}


def render_page_rgb(page: fitz.Page, dpi: int = 150) -> np.ndarray:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    png = pix.tobytes("png")
    img = Image.open(BytesIO(png)).convert("RGB")
    return np.array(img)


def load_publaynet_model():
    return PaddleDetectionLayoutModel(
        config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config",
        threshold=0.5,
        label_map=LABEL_MAP,
        enforce_cpu=True,
        enable_mkldnn=True,
    )



def detect_layout_for_pdf(pdf_path: str, dpi: int = 150) -> List[Dict[str, Any]]:
    model = load_publaynet_model()
    outputs: List[Dict[str, Any]] = []

    with fitz.open(pdf_path) as pdf:
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            image = render_page_rgb(page, dpi=dpi)
            layout = model.detect(image)

            for j, b in enumerate(layout):
                x0, y0, x1, y1 = b.coordinates
                outputs.append(
                    {
                        "page_index": page_index,
                        "region_id": f"r_{page_index}_{j}",
                        "type": b.type,  # Text/Title/List/Table/Figure
                        "score": float(getattr(b, "score", 1.0)),
                        "image_bbox": [float(x0), float(y0), float(x1), float(y1)],
                        "dpi": dpi,
                    }
                )

    return outputs