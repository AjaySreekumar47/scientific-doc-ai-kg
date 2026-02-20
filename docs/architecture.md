# Architecture

## Data Flow
1. **FastAPI** receives PDF upload
2. **Extraction**
   - PyMuPDF native text extraction + block bboxes
   - OCR fallback (Tesseract) for scanned pages
   - Figure detection via PyMuPDF images + bbox
   - Table extraction via Camelot + bbox
   - Caption linking (figure/table ↔ caption block)
3. **Assets**
   - Cropped figure/table PNGs saved under `assets/<doc_id>/...`
4. **Knowledge Graph**
   - RDFLib converts extracted JSON → RDF (Turtle)
   - Uploaded into **GraphDB** as a **named graph per document**
5. **Query**
   - `/sparql` endpoint proxies SPARQL queries to GraphDB
   - Asset endpoints serve cropped provenance images

## Components
- `app/` FastAPI app + pipelines
- `docker-compose.yml` GraphDB container
- `assets/` runtime-generated crops (gitignored)