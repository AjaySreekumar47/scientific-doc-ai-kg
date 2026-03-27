from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
import httpx

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.settings import settings 

from typing import Optional

load_dotenv()

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
async def ingest_to_graphdb(file: UploadFile = File(...), user_goal: Optional[str] = None):
    pdf_path, original_name = await save_upload_temp(file)

    from app.agents.orchestrator import run_agentic_extraction
    from app.pipelines.extract.run import extract_pymupdf_basic
    from app.pipelines.kg.rdf_builder import build_rdf_from_document
    from app.pipelines.kg.graphdb_client import upload_ttl_to_graphdb

    doc_id = f"doc_{uuid.uuid4().hex[:12]}"

    orchestration_result = run_agentic_extraction(
        pdf_path,
        original_name=original_name,
        doc_id=doc_id,
        assets_root=ASSETS_ROOT,
        user_goal=user_goal,
    )

    verified_claims = orchestration_result["verified_claims"]

    # rebuild the extraction doc once for RDF serialization
    doc = extract_pymupdf_basic(
        pdf_path,
        doc_id=doc_id,
        assets_root=ASSETS_ROOT,
    )
    doc.pdf_filename = original_name

    g = build_rdf_from_document(doc, verified_claims=verified_claims)
    ttl = g.serialize(format="turtle")

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
        "num_verified_claims": len(verified_claims),
        "verified_claims": verified_claims,
    }


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


class PlanRequest(BaseModel):
    user_goal: Optional[str] = None
@app.post("/plan")
async def plan(file: UploadFile = File(...), user_goal: Optional[str] = None):
    pdf_path, original_name = await save_upload_temp(file)

    from app.agents.planner import build_execution_plan

    execution_plan = build_execution_plan(pdf_path, user_goal=user_goal)
    execution_plan.document_name = original_name

    return execution_plan

@app.post("/agent_extract")
async def agent_extract(file: UploadFile = File(...), user_goal: Optional[str] = None):
    pdf_path, original_name = await save_upload_temp(file)

    from app.agents.orchestrator import run_agentic_extraction

    doc_id = f"doc_{uuid.uuid4().hex[:12]}"

    result = run_agentic_extraction(
        pdf_path,
        original_name=original_name,
        doc_id=doc_id,
        assets_root=ASSETS_ROOT,
        user_goal=user_goal,
    )

    return result

@app.post("/agent_trace")
async def agent_trace(file: UploadFile = File(...), user_goal: Optional[str] = None):
    pdf_path, original_name = await save_upload_temp(file)

    from app.agents.orchestrator import run_agentic_extraction

    doc_id = f"doc_{uuid.uuid4().hex[:12]}"

    result = run_agentic_extraction(
        pdf_path,
        original_name=original_name,
        doc_id=doc_id,
        assets_root=ASSETS_ROOT,
        user_goal=user_goal,
    )

    layout_out = result["agent_outputs"]["layout_block_agent"]
    figtab_out = result["agent_outputs"]["figure_table_agent"]
    sem_out = result["agent_outputs"]["semantic_extraction_agent"]
    ver_out = result["agent_outputs"]["verification_agent"]

    verified_claims = ver_out["verified_claims"]

    num_verified = sum(1 for c in verified_claims if c["verification_status"] == "verified")
    num_weak = sum(1 for c in verified_claims if c["verification_status"] == "weak")
    num_rejected = sum(1 for c in verified_claims if c["verification_status"] == "rejected")

    return {
        "document_id": result["document_id"],
        "document_name": result["document_name"],
        "execution_plan": result["execution_plan"],
        "agent_trace": [
            {
                "agent_name": "planner_agent",
                "status": "completed",
                "notes": result["execution_plan"].get("notes"),
            },
            {
                "agent_name": "layout_block_agent",
                "status": "completed",
                "num_blocks": layout_out["num_blocks"],
            },
            {
                "agent_name": "figure_table_agent",
                "status": "completed",
                "num_figures": figtab_out["num_figures"],
                "num_tables": figtab_out["num_tables"],
            },
            {
                "agent_name": "semantic_extraction_agent",
                "status": "completed",
                "num_claims": sem_out["num_claims"],
            },
            {
                "agent_name": "verification_agent",
                "status": "completed",
                "num_verified": num_verified,
                "num_weak": num_weak,
                "num_rejected": num_rejected,
            },
            {
                "agent_name": "rdf_graph_builder_agent",
                "status": "ready",
                "num_verified_claims_available": result["num_verified_claims"],
            },
        ],
        "summary": {
            "num_blocks": layout_out["num_blocks"],
            "num_figures": figtab_out["num_figures"],
            "num_tables": figtab_out["num_tables"],
            "num_claims": sem_out["num_claims"],
            "num_verified_claims": result["num_verified_claims"],
            "num_weak_claims": num_weak,
            "num_rejected_claims": num_rejected,
        },
    }

class AskGraphRequest(BaseModel):
    doc_id: str
    question: str

@app.post("/ask_graph")
async def ask_graph(req: AskGraphRequest):
    from app.agents.query_agent import run_query_orchestrator_agent

    result = run_query_orchestrator_agent(
        question=req.question,
        doc_id=req.doc_id,
        graphdb_base_url=GRAPHDB_BASE_URL,
        repo_id=GRAPHDB_REPO_ID,
    )

    return result