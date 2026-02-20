from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict

from typing import Optional


class BlockType(str, Enum):
    title = "title"
    heading = "heading"
    text = "text"
    caption = "caption"
    table = "table"
    figure = "figure"


class BBoxNorm(BaseModel):
    """
    Normalized bounding box in page coordinates: values in [0, 1].
    (x0, y0) is top-left, (x1, y1) is bottom-right.
    """
    model_config = ConfigDict(extra="forbid")

    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pdf_filename: str
    page_index: int = Field(ge=0)
    bbox_norm: BBoxNorm
    extraction_method: str  # e.g., "pymupdf", "pdfplumber", "layoutparser", "camelot", "ocr"
    notes: Optional[str] = None


class DocumentMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: Optional[str] = None
    authors: Optional[List[str]] = None
    doi: Optional[str] = None
    year: Optional[int] = None


class PageInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_index: int = Field(ge=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)


class Block(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str
    type: BlockType
    page_index: int = Field(ge=0)
    bbox_norm: BBoxNorm
    text: Optional[str] = None

    # Optional linkage (caption <-> figure/table, etc.)
    ref_ids: List[str] = Field(default_factory=list)

    provenance: Provenance


class TableCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    text: str = ""


class TableObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str
    page_index: int = Field(ge=0)
    bbox_norm: BBoxNorm
    caption_block_id: Optional[str] = None
    cells: List[TableCell] = Field(default_factory=list)
    provenance: Provenance
    image_path: Optional[str] = None


class FigureObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    figure_id: str
    page_index: int = Field(ge=0)
    bbox_norm: BBoxNorm
    caption_block_id: Optional[str] = None
    image_path: Optional[str] = None  # relative path under artifacts/
    provenance: Provenance


class ExtractionDocument(BaseModel):
    """
    Canonical schema output of the extraction pipeline.
    """
    model_config = ConfigDict(extra="forbid")

    document_id: str
    pdf_filename: str
    pdf_sha256: str

    meta: DocumentMeta = Field(default_factory=DocumentMeta)
    pages: List[PageInfo] = Field(default_factory=list)

    blocks: List[Block] = Field(default_factory=list)
    tables: List[TableObject] = Field(default_factory=list)
    figures: List[FigureObject] = Field(default_factory=list)

    # Room for future pipeline annotations (NER, embeddings, etc.)
    annotations: Dict[str, Any] = Field(default_factory=dict)