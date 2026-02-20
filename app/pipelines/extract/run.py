from __future__ import annotations

import hashlib
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps

from app.pipelines.extract.schema import (
    BBoxNorm,
    Block,
    BlockType,
    ExtractionDocument,
    FigureObject,
    PageInfo,
    Provenance,
    TableCell,
    TableObject,
)


# ----------------------------
# Helpers
# ----------------------------

HEADING_WORDS = {
    "education",
    "experience",
    "projects",
    "skills",
    "technical skills",
    "publications",
    "certifications",
    "summary",
    "objective",
}

def is_table_caption(text: str) -> bool:
    t_one = re.sub(r"\s+", " ", (text or "")).strip()
    return re.match(r"^(table)\s*\d+[:.\-]?\s+", t_one, flags=re.IGNORECASE) is not None

def save_clip_png(
    page: fitz.Page,
    clip: fitz.Rect,
    out_path: Path,
    dpi: int = 200,
) -> bool:
    """
    Render a clipped region to PNG. Returns True if saved.
    """
    if clip.is_empty or clip.width <= 2 or clip.height <= 2:
        return False

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    if pix.width <= 2 or pix.height <= 2:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    return True

def extract_tables_with_camelot(
    pdf_path: Path,
    pdf_filename: str,
    doc_id: str,
    assets_root: Path | None,
    dpi: int = 200,
) -> list[TableObject]:
    """
    Best-effort table extraction using Camelot (stream flavor).
    Also saves a cropped PNG for each detected table (if assets_root provided).
    """
    try:
        import camelot
    except Exception:
        return []

    tables_out: list[TableObject] = []

    # 1) Run Camelot
    try:
        c_tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
    except Exception:
        return []

    # 2) Keep the PDF open for the entire cropping loop (critical)
    try:
        pdf = fitz.open(pdf_path)
    except Exception:
        pdf = None

    try:
        for t_idx, t in enumerate(c_tables):
            # page index (camelot is 1-based)
            try:
                page_index = int(t.page) - 1
            except Exception:
                continue

            # bbox from camelot
            try:
                x1, y1, x2, y2 = t._bbox
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            except Exception:
                continue

            # load page from the OPEN pdf
            if pdf is None or page_index < 0 or page_index >= len(pdf):
                continue
            page = pdf.load_page(page_index)
            rect = page.rect
            page_w = float(rect.width)
            page_h = float(rect.height)

            bbox = norm_bbox(x1, y1, x2, y2, page_w, page_h)

            # cells as TableCell list
            cells: list[TableCell] = []
            try:
                df = t.df.astype(str)
                n_rows, n_cols = df.shape
                for r in range(n_rows):
                    for c in range(n_cols):
                        val = (df.iat[r, c] or "").strip()
                        cells.append(
                            TableCell(
                                row=r,
                                col=c,
                                text=val,
                                bbox_norm=None,
                                confidence=None,
                            )
                        )
            except Exception:
                cells = []

            # Optional crop-save
            image_path_str: str | None = None
            if assets_root is not None:
                out_dir = assets_root / doc_id / "tables"
                table_id = f"tbl_{page_index}_{t_idx}"
                out_path = out_dir / f"{table_id}.png"

                # Two coordinate interpretations + clamp to page bounds
                rect_a = fitz.Rect(x1, y1, x2, y2) & page.rect
                rect_b = fitz.Rect(
                    x1,
                    page_h - y2,
                    x2,
                    page_h - y1,
                ) & page.rect

                saved = save_clip_png(page, rect_a, out_path, dpi=dpi)
                if not saved:
                    saved = save_clip_png(page, rect_b, out_path, dpi=dpi)

                if saved:
                    image_path_str = str(out_path.relative_to(assets_root))

            prov = Provenance(
                pdf_filename=pdf_filename,
                page_index=page_index,
                bbox_norm=bbox,
                extraction_method="camelot_stream_v1",
                notes=f"camelot table index={t_idx}; saved_clip={bool(image_path_str)}; bbox_raw={getattr(t, '_bbox', None)}",
            )

            # construct TableObject (skip invalid tables rather than crashing)
            try:
                tables_out.append(
                    TableObject(
                        table_id=f"tbl_{page_index}_{t_idx}",
                        page_index=page_index,
                        bbox_norm=bbox,
                        cells=cells,
                        caption_block_id=None,
                        image_path=image_path_str,
                        provenance=prov,
                    )
                )
            except Exception:
                continue

    finally:
        if pdf is not None:
            pdf.close()

    return tables_out


def bbox_center(b: BBoxNorm) -> tuple[float, float]:
    return ((b.x0 + b.x1) / 2.0, (b.y0 + b.y1) / 2.0)

def vertical_gap(upper: BBoxNorm, lower: BBoxNorm) -> float:
    # assumes y increases downward
    return lower.y0 - upper.y1

def link_captions_to_figures(
    blocks: list[Block],
    figures: list[FigureObject],
    max_gap: float = 0.12,  # normalized page height
) -> None:
    # Group by page
    blocks_by_page: dict[int, list[Block]] = {}
    figs_by_page: dict[int, list[FigureObject]] = {}

    for b in blocks:
        blocks_by_page.setdefault(b.page_index, []).append(b)
    for f in figures:
        figs_by_page.setdefault(f.page_index, []).append(f)

    for page_idx, page_blocks in blocks_by_page.items():
        caps = [b for b in page_blocks if b.type == BlockType.caption]
        figs = figs_by_page.get(page_idx, [])
        if not caps or not figs:
            continue

        # For each caption, pick nearest figure above it (preferred),
        # else nearest by center distance.
        for cap in caps:
            best = None
            best_score = 1e9

            for fig in figs:
                gap = vertical_gap(fig.bbox_norm, cap.bbox_norm)

                # Prefer figure above caption with small positive gap
                if 0.0 <= gap <= max_gap:
                    score = gap
                else:
                    # Fallback: center distance
                    cx1, cy1 = bbox_center(fig.bbox_norm)
                    cx2, cy2 = bbox_center(cap.bbox_norm)
                    score = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5 + 1.0  # penalize non-above

                if score < best_score:
                    best_score = score
                    best = fig

            if best is not None and best.caption_block_id is None:
                best.caption_block_id = cap.block_id
                # store reverse link in caption block (figure id)
                if best.figure_id not in cap.ref_ids:
                    cap.ref_ids.append(best.figure_id)

def link_captions_to_tables(
    blocks: list[Block],
    tables: list[TableObject],
    max_gap: float = 0.15,  # normalized page height
) -> None:
    blocks_by_page: dict[int, list[Block]] = {}
    tables_by_page: dict[int, list[TableObject]] = {}

    for b in blocks:
        blocks_by_page.setdefault(b.page_index, []).append(b)
    for t in tables:
        tables_by_page.setdefault(t.page_index, []).append(t)

    for page_idx, page_blocks in blocks_by_page.items():
        caps = [b for b in page_blocks if b.type == BlockType.caption and is_table_caption(b.text or "")]
        tbls = tables_by_page.get(page_idx, [])
        if not caps or not tbls:
            continue

        for cap in caps:
            best = None
            best_score = 1e9

            for tbl in tbls:
                gap = vertical_gap(tbl.bbox_norm, cap.bbox_norm)

                # Prefer table above caption
                if 0.0 <= gap <= max_gap:
                    score = gap
                else:
                    cx1, cy1 = bbox_center(tbl.bbox_norm)
                    cx2, cy2 = bbox_center(cap.bbox_norm)
                    score = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5 + 1.0

                if score < best_score:
                    best_score = score
                    best = tbl

            if best is not None and best.caption_block_id is None:
                best.caption_block_id = cap.block_id
                if best.table_id not in cap.ref_ids:
                    cap.ref_ids.append(best.table_id)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_bbox(x0: float, y0: float, x1: float, y1: float, page_w: float, page_h: float) -> BBoxNorm:
    nx0 = max(0.0, min(1.0, x0 / page_w))
    ny0 = max(0.0, min(1.0, y0 / page_h))
    nx1 = max(0.0, min(1.0, x1 / page_w))
    ny1 = max(0.0, min(1.0, y1 / page_h))
    return BBoxNorm(x0=nx0, y0=ny0, x1=nx1, y1=ny1)


def classify_block_text(text: str, page_index: int, bbox: BBoxNorm) -> BlockType:
    t = (text or "").strip()
    t_low = re.sub(r"\s+", " ", t.lower()).strip()

    # Title heuristic: page 0, near top, not too long
    if page_index == 0 and bbox.y0 < 0.15 and len(t) <= 140 and ("\n" in t or len(t.split()) <= 7):
        return BlockType.title

    # Caption heuristic: "Figure 1", "Fig. 2", "Table 3", etc.
    t_one = re.sub(r"\s+", " ", t).strip()
    if re.match(r"^(fig(ure)?\.?|table)\s*\d+[:.\-]?\s+", t_one, flags=re.IGNORECASE):
        return BlockType.caption

    # Heading heuristic: short and section-like
    words = t_low.split()
    if 1 <= len(words) <= 5:
        if t_low in HEADING_WORDS:
            return BlockType.heading
        if t.isupper() and len(words) <= 6:
            return BlockType.heading
        if len(words) == 1 and t and t[0].isalpha():
            return BlockType.heading

    return BlockType.text


def ocr_page_with_tesseract(page: fitz.Page, dpi: int = 300) -> str:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    png_bytes = pix.tobytes("png")
    img = Image.open(BytesIO(png_bytes))

    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.point(lambda p: 255 if p > 170 else 0)

    # psm 3 = auto page segmentation
    text = pytesseract.image_to_string(img, config="--psm 3")
    return text.strip()


def extract_figure_regions_from_images(
    pdf: fitz.Document,
    page: fitz.Page,
    page_index: int,
    pdf_filename: str,
    page_w: float,
    page_h: float,
    doc_id: str,
    assets_root: Path | None,
    dpi: int = 200,
) -> list[FigureObject]:
    figs: list[FigureObject] = []

    fig_dir: Path | None = None
    if assets_root is not None:
        fig_dir = assets_root / doc_id / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

    for img_i, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        rects = page.get_image_rects(xref)
        if not rects:
            continue

        for r_i, r in enumerate(rects):
            bbox = norm_bbox(r.x0, r.y0, r.x1, r.y1, page_w, page_h)

            figure_id = f"fig_{page_index}_img_{img_i}_{r_i}"
            image_path_str: str | None = None

            # Save cropped image if assets_root provided
            if fig_dir is not None:
                # Render clipped region at dpi
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, clip=r, alpha=False)

                out_path = fig_dir / f"{figure_id}.png"
                pix.save(str(out_path))

                # Store relative path (portable)
                image_path_str = str(out_path.relative_to(assets_root))

            prov = Provenance(
                pdf_filename=pdf_filename,
                page_index=page_index,
                bbox_norm=bbox,
                extraction_method="pymupdf_get_images_v1",
                notes=f"Image xref={xref}; saved_clip={bool(image_path_str)}",
            )

            figs.append(
                FigureObject(
                    figure_id=figure_id,
                    page_index=page_index,
                    bbox_norm=bbox,
                    caption_block_id=None,
                    image_path=image_path_str,
                    provenance=prov,
                )
            )

    return figs



# ----------------------------
# Main extractor (heuristic B)
# ----------------------------

def extract_pymupdf_basic(pdf_path: Path, *, doc_id: str | None = None, assets_root: Path | None = None) -> ExtractionDocument:
    """
    Heuristic layout-aware extractor (Windows-stable):
      - Native text blocks with geometry via page.get_text("blocks")
      - OCR fallback for scanned pages
      - Dedup blocks
      - Title/Heading/Caption/Text classification (simple)
      - Figure regions via non-text blocks from get_text("blocks")
    """
    doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
    pdf_hash = sha256_file(pdf_path)

    pages: list[PageInfo] = []
    blocks: list[Block] = []
    figures: list[FigureObject] = []
    page_stats: list[dict] = []
    tables: list[TableObject] = []

    with fitz.open(pdf_path) as pdf:
        for i in range(pdf.page_count):
            page = pdf.load_page(i)

            rect = page.rect
            page_w = float(rect.width)
            page_h = float(rect.height)

            page_figs = extract_figure_regions_from_images(
                    pdf=pdf,
                    page=page,
                    page_index=i,
                    pdf_filename=pdf_path.name,
                    page_w=page_w,
                    page_h=page_h,
                    doc_id=doc_id,
                    assets_root=assets_root,
                    dpi=200,
                )

            figures.extend(page_figs)


            pages.append(PageInfo(page_index=i, width_px=int(page_w), height_px=int(page_h)))

            # 1) Native text extraction (for scanned detection)
            native_text = page.get_text("text").strip()
            native_text_len = len(native_text)
            native_words = len(native_text.split()) if native_text else 0

            is_probably_scanned = native_text_len < 20

            # OCR bookkeeping
            ocr_used = False
            ocr_text_len = 0
            ocr_error: Optional[str] = None

            # For scanned pages we produce one full-page OCR text block (no bboxes)
            if is_probably_scanned:
                final_text = ""
                extraction_method = "pymupdf_text_v1"
                notes = "Scanned heuristic triggered; attempting OCR."

                try:
                    ocr_text = ocr_page_with_tesseract(page, dpi=300)
                    ocr_text_len = len(ocr_text)
                    if ocr_text:
                        final_text = ocr_text
                        ocr_used = True
                        extraction_method = "tesseract_ocr_v1"
                        notes = "OCR-derived full-page text block (scanned-page fallback)."
                except Exception as e:
                    ocr_error = f"{type(e).__name__}: {e}"
                    notes = f"OCR failed: {ocr_error}"

                if final_text:
                    bbox = BBoxNorm(x0=0.0, y0=0.0, x1=1.0, y1=1.0)
                    prov = Provenance(
                        pdf_filename=pdf_path.name,
                        page_index=i,
                        bbox_norm=bbox,
                        extraction_method=extraction_method,
                        notes=notes,
                    )
                    blocks.append(
                        Block(
                            block_id=f"b_{i}_ocr_fulltext",
                            type=BlockType.text,
                            page_index=i,
                            bbox_norm=bbox,
                            text=final_text,
                            ref_ids=[],
                            provenance=prov,
                        )
                    )

                final_len = len(final_text)
                final_words = len(final_text.split()) if final_text else 0

                page_stats.append(
                    {
                        "page_index": i,
                        "native_text_len": native_text_len,
                        "native_num_words": native_words,
                        "is_probably_scanned": is_probably_scanned,
                        "ocr_used": ocr_used,
                        "ocr_text_len": ocr_text_len,
                        "ocr_error": ocr_error,
                        "final_text_len": final_len,
                        "final_num_words": final_words,
                        "num_text_blocks": 0,
                        "num_figures": 0,
                    }
                )
                continue  # done with this page

            # 2) Native page blocks with geometry (text + non-text)
            text_blocks = page.get_text("blocks")

            # Text blocks
            seen = set()
            page_text_block_count = 0
            block_idx = 0

            for tb in text_blocks:
                x0, y0, x1, y1, t, block_no, block_type = tb

                if block_type != 0:
                    continue  # handle non-text below

                txt = (t or "").strip()
                if not txt:
                    continue

                bbox = norm_bbox(x0, y0, x1, y1, page_w, page_h)

                # Dedup key: page + rounded bbox + text hash
                bbox_key = (round(bbox.x0, 4), round(bbox.y0, 4), round(bbox.x1, 4), round(bbox.y1, 4))
                txt_key = hash(txt[:500])
                key = (i, bbox_key, txt_key)
                if key in seen:
                    continue
                seen.add(key)

                btype = classify_block_text(txt, i, bbox)

                prov = Provenance(
                    pdf_filename=pdf_path.name,
                    page_index=i,
                    bbox_norm=bbox,
                    extraction_method="pymupdf_blocks_v1",
                    notes="PyMuPDF text block bbox (deduped).",
                )

                blocks.append(
                    Block(
                        block_id=f"b_{i}_{block_idx}",
                        type=btype,
                        page_index=i,
                        bbox_norm=bbox,
                        text=txt,
                        ref_ids=[],
                        provenance=prov,
                    )
                )
                block_idx += 1
                page_text_block_count += 1

            # Figure/image blocks (non-text blocks)
            fig_idx = 0
            for tb in text_blocks:
                x0, y0, x1, y1, t, block_no, block_type = tb
                if block_type == 0:
                    continue

                bbox = norm_bbox(x0, y0, x1, y1, page_w, page_h)
                prov = Provenance(
                    pdf_filename=pdf_path.name,
                    page_index=i,
                    bbox_norm=bbox,
                    extraction_method="pymupdf_image_block_v1",
                    notes=f"Non-text block from PyMuPDF blocks (block_type={block_type}).",
                )

                figures.append(
                    FigureObject(
                        figure_id=f"fig_{i}_{fig_idx}",
                        page_index=i,
                        bbox_norm=bbox,
                        caption_block_id=None,
                        image_path=None,
                        provenance=prov,
                    )
                )
                fig_idx += 1

            page_stats.append(
                {
                    "page_index": i,
                    "native_text_len": native_text_len,
                    "native_num_words": native_words,
                    "is_probably_scanned": is_probably_scanned,
                    "ocr_used": False,
                    "ocr_text_len": 0,
                    "ocr_error": None,
                    "final_text_len": native_text_len,
                    "final_num_words": native_words,
                    "num_text_blocks": page_text_block_count,
                    "num_figures": len(page_figs),
                }
            )

    # Basic counts for convenience
    type_counts = {"title": 0, "heading": 0, "caption": 0, "text": 0}

    captions = [
                    {"block_id": b.block_id, "page_index": b.page_index, "text": (b.text or "")[:120]}
                    for b in blocks
                    if b.type == BlockType.caption
                ]
    

    for b in blocks:
        if b.type.value in type_counts:
            type_counts[b.type.value] += 1
        
    tables = extract_tables_with_camelot(
            pdf_path,
            pdf_path.name,
            doc_id=doc_id,
            assets_root=assets_root,
            dpi=200,
        )
    
    link_captions_to_figures(blocks, figures)
    linked = sum(1 for f in figures if f.caption_block_id is not None)

    link_captions_to_tables(blocks, tables)
    num_tables_with_captions = sum(1 for t in tables if t.caption_block_id is not None)

    return ExtractionDocument(
        document_id=doc_id,
        pdf_filename=pdf_path.name,
        pdf_sha256=pdf_hash,
        pages=pages,
        blocks=blocks,
        tables=tables,
        figures=figures,
        annotations={
            "extractor": "heuristic_layout_v1",
            "num_pages": len(pages),
            "num_blocks": len(blocks),
            "num_figures": len(figures),
            "type_counts": type_counts,
            "num_captions": len(captions),
            "captions_found": captions[:25],  # cap for response size
            "num_figures_with_captions": linked,
            "num_tables_with_captions": num_tables_with_captions,
            "num_tables": len(tables),
            "page_stats": page_stats,
        },
    )