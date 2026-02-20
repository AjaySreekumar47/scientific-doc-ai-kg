import hashlib
from app.pipelines.extract.schema import (
    ExtractionDocument, PageInfo, BBoxNorm, Provenance, Block, BlockType
)

dummy_pdf = "dummy.pdf"
doc = ExtractionDocument(
    document_id="doc_dummy_001",
    pdf_filename=dummy_pdf,
    pdf_sha256=hashlib.sha256(b"dummy").hexdigest(),
    pages=[PageInfo(page_index=0, width_px=1000, height_px=1000)],
    blocks=[
        Block(
            block_id="b0",
            type=BlockType.text,
            page_index=0,
            bbox_norm=BBoxNorm(x0=0.1, y0=0.1, x1=0.9, y1=0.2),
            text="hello world",
            provenance=Provenance(
                pdf_filename=dummy_pdf,
                page_index=0,
                bbox_norm=BBoxNorm(x0=0.1, y0=0.1, x1=0.9, y1=0.2),
                extraction_method="smoke_test",
            ),
        )
    ],
)

print(doc.model_dump_json(indent=2))