# Autonomous Scientific Knowledge Graph Agent

![Python](https://img.shields.io/badge/Python-3.13%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![GraphDB](https://img.shields.io/badge/GraphDB-knowledge%20graph-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A multi-agent scientific document intelligence system that ingests research PDFs, extracts layout-aware content with provenance, validates semantic claims, builds a knowledge graph, and answers natural-language research questions through graph-grounded reasoning.

## What it does

This system transforms scientific PDFs into an explainable, queryable research knowledge graph.

Core capabilities:

- Planner-driven execution
- Layout-aware block extraction
- Figure/table detection with cropped provenance assets
- OCR fallback for scanned PDFs
- Semantic claim extraction
- Verification of claims against source evidence
- RDF graph construction and GraphDB ingestion
- Natural-language question answering via SPARQL-backed retrieval
- Evidence-backed responses with provenance

## Agent Architecture

```text
User uploads PDF(s)
        ↓
Planner Agent
        ↓
Parallel extraction stage
   ├─ Layout & Block Agent
   ├─ Figure/Table Agent
   └─ Semantic Extraction Agent
        ↓
Verification Agent
        ↓
RDF / Graph Builder Agent
        ↓
GraphDB
        ↓
Query Orchestrator Agent
   ├─ SPARQL generation
   ├─ Graph retrieval
   └─ Answer synthesis
        ↓
Explainable response + provenance
````

## Agent Roles

### 1. Planner Agent

Determines how the document should be processed.

Decides:

* whether OCR is needed
* which agents should run
* whether the system operates in single-document mode

### 2. Layout & Block Agent

Uses the extraction stack to recover:

* text blocks
* headings
* captions
* normalized bounding boxes
* page provenance

### 3. Figure/Table Agent

Recovers:

* figures
* tables
* caption links
* cropped figure/table assets

### 4. Semantic Extraction Agent

Derives lightweight scientific meaning from extracted content, such as:

* methods
* datasets
* metrics
* result claims
* limitations
* future work indicators

### 5. Verification Agent

Checks whether extracted claims are grounded in source evidence.

Outputs:

* verified
* weak
* rejected

Only verified claims are committed into the graph.

### 6. Query Orchestrator Agent

Converts natural-language questions into graph-grounded answers.

Flow:

* classify question
* generate SPARQL
* retrieve graph evidence
* synthesize answer
* return supporting provenance

## System Workflow

```text
PDF
→ Planner Agent
→ Extraction Agents
→ Semantic Extraction Agent
→ Verification Agent
→ Verified Claim Graph Commit
→ GraphDB
→ Query Orchestrator Agent
→ Answer + Evidence
```

## Main Features

### Layout-Aware Extraction

* PyMuPDF native text extraction
* OCR fallback using Tesseract
* normalized bounding boxes
* caption detection and linking
* figure and table region extraction

### Provenance Assets

* cropped figure PNGs
* cropped table PNGs
* page-aware, bbox-linked provenance

### Knowledge Graph Construction

* RDFLib graph building
* named graph per document
* GraphDB ingestion
* SPARQL querying

### Agentic Reasoning

* visible planner decisions
* structured agent outputs
* validation loop
* verified-only claim insertion
* graph-backed question answering

## API Endpoints

### Core Pipeline

* `POST /plan` — generate execution plan
* `POST /agent_extract` — run planner + extraction agents
* `POST /agent_trace` — return orchestration summary
* `POST /extract` — structured extraction output
* `POST /extract_to_rdf` — RDF/Turtle output
* `POST /ingest_to_graphdb` — ingest verified graph into GraphDB
* `POST /sparql` — execute raw SPARQL
* `POST /ask_graph` — ask natural-language questions over the graph

### Evidence Assets

* `GET /docs/{doc_id}/figures/{figure_id}/image`
* `GET /docs/{doc_id}/tables/{table_id}/image`

## Streamlit Demo

The project includes a lightweight Streamlit demo surface with four tabs:

* **Ingest**
* **Agent Trace**
* **Ask Graph**
* **Evidence Viewer**

Run it with:

```powershell
streamlit run streamlit_app.py
```

## Quickstart

### 1. Start GraphDB

```bash
docker compose up -d
```

GraphDB UI:
`http://localhost:7200`

Create a repository:

* Repository ID: `scientific-docs`
* Ruleset: `No inference`

### 2. Create environment

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Create `.env` from `.env.example` or define:

```env
GRAPHDB_BASE_URL=http://localhost:7200
GRAPHDB_REPO_ID=scientific-docs
ASSETS_ROOT=assets
```

### 5. Run FastAPI

Windows:

```powershell
.\scripts\dev.ps1
```

Linux/Mac:

```bash
bash scripts/dev.sh
```

Swagger:
`http://127.0.0.1:8000/docs`

### 6. Run Streamlit

```powershell
streamlit run streamlit_app.py
```

## Example Demo Flow

1. Upload a scientific PDF
2. Inspect Planner output using `/plan`
3. Run `/agent_trace` to view orchestration
4. Ingest verified claims into GraphDB with `/ingest_to_graphdb`
5. Ask a question through `/ask_graph`
6. Inspect figure/table evidence through the evidence viewer

## Example Questions

* What datasets and metrics were used?
* What is the main result?
* What figures are present?
* What tables are present?
* Show me evidence for the main finding.

## Example SPARQL

List documents:

```sparql
PREFIX ex: <http://example.org/schema/>
SELECT ?d ?fn
WHERE {
  ?d a ex:Document ;
     ex:filename ?fn .
}
LIMIT 25
```

Query verified claims in a document graph:

```sparql
PREFIX ex: <http://example.org/schema/>
SELECT ?claim ?text ?ctype ?status ?conf
WHERE {
  GRAPH <http://example.org/graph/DOC_ID_HERE> {
    ?claim a ex:Claim ;
           ex:claimText ?text ;
           ex:claimType ?ctype ;
           ex:verificationStatus ?status ;
           ex:confidence ?conf .
  }
}
LIMIT 25
```

## Repository Structure

```text
app/
  agents/
  core/
  pipelines/
scripts/
docs/
streamlit_app.py
docker-compose.yml
requirements.txt
requirements-dev.txt
```

## Notes

* `assets/` is generated at runtime and should not be committed
* GraphDB runs through Docker
* the system uses a Windows-stable heuristic layout pipeline instead of LayoutParser
* verified claims are treated as first-class graph entities

## Skills Demonstrated

* agent orchestration
* document intelligence
* OCR and layout-aware extraction
* semantic extraction
* self-validation loops
* RDF / knowledge graph construction
* GraphDB / SPARQL
* provenance-backed retrieval
* API design with FastAPI
* Streamlit demo development
* CI workflow integration

## License

MIT