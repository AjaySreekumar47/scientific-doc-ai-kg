from __future__ import annotations

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

from app.pipelines.extract.schema import ExtractionDocument


# Namespaces
EX = Namespace("http://example.org/schema/")
DOC = Namespace("http://example.org/doc/")
PROV = Namespace("http://www.w3.org/ns/prov#")


def build_rdf_from_document(doc: ExtractionDocument) -> Graph:
    g = Graph()
    g.bind("ex", EX)
    g.bind("doc", DOC)
    g.bind("prov", PROV)

    doc_uri = DOC[f"document/{doc.document_id}"]
    g.add((doc_uri, RDF.type, EX.Document))
    g.add((doc_uri, EX.filename, Literal(doc.pdf_filename)))
    g.add((doc_uri, EX.sha256, Literal(doc.pdf_sha256)))

    # Pages
    for p in doc.pages:
        page_uri = DOC[f"{doc.document_id}/page/{p.page_index}"]
        g.add((page_uri, RDF.type, EX.Page))
        g.add((page_uri, EX.pageIndex, Literal(p.page_index, datatype=XSD.integer)))
        g.add((doc_uri, EX.hasPage, page_uri))

    # Blocks
    for b in doc.blocks:
        block_uri = DOC[f"{doc.document_id}/block/{b.block_id}"]
        page_uri = DOC[f"{doc.document_id}/page/{b.page_index}"]

        g.add((block_uri, RDF.type, EX.Block))
        g.add((block_uri, EX.blockType, Literal(b.type.value)))
        g.add((block_uri, EX.text, Literal(b.text or "")))

        g.add((block_uri, EX.x0, Literal(b.bbox_norm.x0)))
        g.add((block_uri, EX.y0, Literal(b.bbox_norm.y0)))
        g.add((block_uri, EX.x1, Literal(b.bbox_norm.x1)))
        g.add((block_uri, EX.y1, Literal(b.bbox_norm.y1)))

        g.add((page_uri, EX.hasBlock, block_uri))

    # Figures
    for f in doc.figures:
        fig_uri = DOC[f"{doc.document_id}/figure/{f.figure_id}"]
        page_uri = DOC[f"{doc.document_id}/page/{f.page_index}"]

        g.add((fig_uri, RDF.type, EX.Figure))
        g.add((fig_uri, EX.x0, Literal(f.bbox_norm.x0)))
        g.add((fig_uri, EX.y0, Literal(f.bbox_norm.y0)))
        g.add((fig_uri, EX.x1, Literal(f.bbox_norm.x1)))
        g.add((fig_uri, EX.y1, Literal(f.bbox_norm.y1)))

        g.add((page_uri, EX.hasFigure, fig_uri))

        if f.caption_block_id:
            caption_uri = DOC[f"{doc.document_id}/block/{f.caption_block_id}"]
            g.add((fig_uri, EX.hasCaptionBlock, caption_uri))

    # Tables
    for t in doc.tables:
        tbl_uri = DOC[f"{doc.document_id}/table/{t.table_id}"]
        page_uri = DOC[f"{doc.document_id}/page/{t.page_index}"]

        g.add((tbl_uri, RDF.type, EX.Table))
        g.add((tbl_uri, EX.x0, Literal(t.bbox_norm.x0)))
        g.add((tbl_uri, EX.y0, Literal(t.bbox_norm.y0)))
        g.add((tbl_uri, EX.x1, Literal(t.bbox_norm.x1)))
        g.add((tbl_uri, EX.y1, Literal(t.bbox_norm.y1)))

        g.add((page_uri, EX.hasTable, tbl_uri))

        if t.caption_block_id:
            caption_uri = DOC[f"{doc.document_id}/block/{t.caption_block_id}"]
            g.add((tbl_uri, EX.hasCaptionBlock, caption_uri))

        # Cells
        for cell in t.cells:
            cell_uri = DOC[f"{doc.document_id}/cell/{t.table_id}_{cell.row}_{cell.col}"]
            g.add((cell_uri, RDF.type, EX.TableCell))
            g.add((cell_uri, EX.row, Literal(cell.row, datatype=XSD.integer)))
            g.add((cell_uri, EX.col, Literal(cell.col, datatype=XSD.integer)))
            g.add((cell_uri, EX.text, Literal(cell.text or "")))

            g.add((tbl_uri, EX.hasCell, cell_uri))

    return g