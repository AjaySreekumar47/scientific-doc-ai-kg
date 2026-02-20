import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File
import httpx
import uuid

from pathlib import Path
import tempfile
from fastapi.responses import FileResponse

# from app.pipelines.extract.run import extract_pymupdf_basic
from app.core.settings import settings

# from app.pipelines.kg.rdf_builder import build_rdf_from_document

# from app.pipelines.kg.graphdb_client import upload_ttl_to_graphdb, sparql_query_graphdb

GRAPHDB_BASE_URL = os.getenv("GRAPHDB_BASE_URL", "http://localhost:7200")
GRAPHDB_REPO_ID = os.getenv("GRAPHDB_REPO_ID", "scientific-docs")
ASSETS_ROOT = Path(os.getenv("ASSETS_ROOT", "assets")).resolve()
ASSETS_ROOT.mkdir(parents=True, exist_ok=True)

async def save_upload_temp(file: UploadFile) -> tuple[Path, str]:
    suffix = Path(file.filename).suffix or ".pdf"
    original_name = file.filename

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    return tmp_path, original_name


app = FastAPI(title="Scientific Document AI + Knowledge Graph")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/graphdb/ping")
async def graphdb_ping():
    """
    Sanity check that the API can reach GraphDB.
    """
    url = settings.graphdb_base_url.rstrip("/") + "/"

    headers = {
        # GraphDB can return 406 if Accept isn't something it likes
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    async with httpx.AsyncClient(timeout=5.0, headers=headers) as client:
        r = await client.get(url)

    return {
        "graphdb_base_url": settings.graphdb_base_url,
        "http_status": r.status_code,
    }


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    pdf_path, original_name = await save_upload_temp(file)

    # ✅ lazy import so CI doesn't need fitz/camelot/tesseract installed
    from app.pipelines.extract.run import extract_pymupdf_basic
    import uuid

    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    doc = extract_pymupdf_basic(pdf_path, doc_id=doc_id, assets_root=ASSETS_ROOT)
    doc.pdf_filename = original_name
    return doc


@app.post("/layout_detect")
async def layout_detect(file: UploadFile = File(...)):
    return {
        "error": "layout_detect disabled in .venv (Windows heuristic pipeline in /extract). "
                 "Use /extract for layout-aware JSON."
    }


@app.post("/extract_to_rdf")
async def extract_to_rdf(file: UploadFile = File(...)):
    pdf_path, original_name = await save_upload_temp(file)

    from app.pipelines.kg.rdf_builder import build_rdf_from_document
    from app.pipelines.extract.run import extract_pymupdf_basic
    import uuid

    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    doc = extract_pymupdf_basic(pdf_path, doc_id=doc_id, assets_root=ASSETS_ROOT)
    doc.pdf_filename = original_name  # overwrite temp filename for provenance + RDF
    g = build_rdf_from_document(doc)
    ttl = g.serialize(format="turtle")

    return {"ttl": ttl}

@app.post("/ingest_to_graphdb")
async def ingest_to_graphdb(file: UploadFile = File(...)):
    pdf_path, original_name = await save_upload_temp(file)

    from app.pipelines.extract.run import extract_pymupdf_basic
    from app.pipelines.kg.rdf_builder import build_rdf_from_document
    from app.pipelines.kg.graphdb_client import upload_ttl_to_graphdb

    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    doc = extract_pymupdf_basic(pdf_path, doc_id=doc_id, assets_root=ASSETS_ROOT)
    doc.pdf_filename = original_name  # overwrite temp filename for provenance + RDF

    g = build_rdf_from_document(doc)
    ttl = g.serialize(format="turtle")

    # Put each document in its own named graph (nice for provenance + delete later)
    context = f"http://example.org/graph/{doc.document_id}"

    status = upload_ttl_to_graphdb(
        graphdb_base_url=GRAPHDB_BASE_URL,
        repo_id=GRAPHDB_REPO_ID,
        ttl=ttl,
        context_iri=context,
    )

    return {
        "document_id": doc.document_id,
        "repo_id": GRAPHDB_REPO_ID,
        "context": context,
        "http_status": status,
        "triple_count": len(g),
    }


from pydantic import BaseModel

class SparqlRequest(BaseModel):
    query: str

@app.post("/sparql")
async def sparql(req: SparqlRequest):
    from app.pipelines.kg.graphdb_client import sparql_query_graphdb

    return sparql_query_graphdb(
        graphdb_base_url=GRAPHDB_BASE_URL,
        repo_id=GRAPHDB_REPO_ID,
        sparql=req.query,
    )

@app.get("/docs/{doc_id}/figures/{figure_id}/image")
def get_figure_image(doc_id: str, figure_id: str):
    path = ASSETS_ROOT / doc_id / "figures" / f"{figure_id}.png"
    if not path.exists():
        return {"error": "Figure image not found", "path": str(path)}
    return FileResponse(path)

## doc_abf5b89a84df; fig_1_img_0_0; doc_abf5b89a84df\\figures\\fig_1_img_0_0.png

@app.get("/docs/{doc_id}/tables/{table_id}/image")
def get_table_image(doc_id: str, table_id: str):
    path = ASSETS_ROOT / doc_id / "tables" / f"{table_id}.png"
    if not path.exists():
        return {"error": "Table image not found", "path": str(path)}
    return FileResponse(path)