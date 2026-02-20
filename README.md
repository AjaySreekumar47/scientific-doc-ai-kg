![Python](https://img.shields.io/badge/Python-3.13%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![GraphDB](https://img.shields.io/badge/GraphDB-knowledge%20graph-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

![CI](https://github.com/AjaySreekumar47/scientific-doc-ai-kg/actions/workflows/ci.yml/badge.svg)

# Scientific Document AI + Knowledge Graph Pipeline (Layout-Aware + RDF/SPARQL + Serving)

End-to-end pipeline for scientific PDFs:
- Extract text, captions, figures, and tables with provenance (page + normalized bbox)
- OCR fallback for scanned PDFs (Tesseract)
- Save cropped figure/table images as assets
- Convert extracted structure to RDF (Turtle) using RDFLib
- Ingest into GraphDB (named graph per document)
- Query via SPARQL through a FastAPI endpoint

## Architecture
PDF → Extraction (PyMuPDF + OCR + Camelot) → JSON Schema → RDF (RDFLib) → GraphDB → SPARQL API  
Additionally: Figure/Table crops saved under `assets/<doc_id>/...` and served via API endpoints.

## Requirements
- Python 3.13+
- Docker Desktop
- Tesseract OCR installed (CLI available)
- GraphDB via Docker (compose)

## Quickstart

(Optional) Copy env template:
```bash
cp .env.example .env
```

(Windows users can just create `.env` manually.)

## Demo (5 minutes)

1) Start GraphDB:
```bash
docker compose up -d
```

2) Run API:
```powershell
.\scripts\dev.ps1
```

3) Ingest a PDF:

- Open http://127.0.0.1:8000/docs

- POST /ingest_to_graphdb → upload a research paper PDF

- Copy document_id and (optionally) a figure_id / table_id

4) View a figure crop:
GET /docs/{doc_id}/figures/{figure_id}/image

5) Run a SPARQL query:
POST /sparql with:

```json
{ "query": "PREFIX ex: <http://example.org/schema/> SELECT ?d ?fn WHERE { ?d a ex:Document ; ex:filename ?fn . } LIMIT 10" }
```

### 1) Start GraphDB
```bash
docker compose up -d
````

GraphDB UI: [http://localhost:7200](http://localhost:7200)

Create a repo in GraphDB:

* Repo ID: `scientific-docs`
* Ruleset: No inference

### 2) Install Python deps

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3) Run API

### Run API (recommended)
**Windows (PowerShell):**
```powershell
.\scripts\dev.ps1

**Linux/Mac:**
```bash
bash scripts/dev.sh
```

Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## API Endpoints (core)

* `POST /extract` → JSON extraction output
* `POST /extract_to_rdf` → Turtle in response
* `POST /ingest_to_graphdb` → Extract + RDF + load into GraphDB (named graph per doc)
* `POST /sparql` → Run SPARQL query against GraphDB repo

## Asset Endpoints

* `GET /docs/{doc_id}/figures/{figure_id}/image` → cropped figure PNG
* `GET /docs/{doc_id}/tables/{table_id}/image` → cropped table PNG

## Example SPARQL Queries

List documents:

```sparql
PREFIX ex: <http://example.org/schema/>
SELECT ?d ?fn WHERE { ?d a ex:Document ; ex:filename ?fn . } LIMIT 25
```

Captions in a specific document graph:

```sparql
PREFIX ex: <http://example.org/schema/>
SELECT ?b ?t WHERE {
  GRAPH <http://example.org/graph/DOC_ID_HERE> {
    ?b a ex:Block ; ex:blockType "caption" ; ex:text ?t .
  }
} LIMIT 25
```

## Notes

* `assets/` is generated at runtime and is intentionally not committed.
* LayoutParser/Detectron2 is not used (Windows-friendly heuristic layout pipeline).
* Table bbox cropping uses a dual-coordinate strategy due to PDF coordinate differences.

## License

MIT