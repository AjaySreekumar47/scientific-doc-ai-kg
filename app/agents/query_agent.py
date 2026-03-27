from __future__ import annotations

from typing import Dict, List

from httpx import ConnectError

from app.pipelines.kg.graphdb_client import sparql_query_graphdb


def classify_question(question: str) -> str:
    q = question.lower()

    if any(word in q for word in ["dataset", "datasets", "benchmark", "corpus"]):
        return "dataset_query"

    if any(
        word in q
        for word in [
            "metric",
            "metrics",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "bleu",
            "rouge",
            "dice",
            "iou",
        ]
    ):
        return "metric_query"

    if any(word in q for word in ["figure", "fig", "image"]):
        return "figure_query"

    if any(word in q for word in ["table", "tables"]):
        return "table_query"

    if any(word in q for word in ["result", "performance", "main finding", "conclusion"]):
        return "result_query"

    return "general_caption_query"


def generate_sparql(question_type: str, doc_id: str) -> str:
    graph_iri = f"http://example.org/graph/{doc_id}"

    if question_type == "dataset_query":
        return f"""
PREFIX ex: <http://example.org/schema/>

SELECT ?block ?text
WHERE {{
  GRAPH <{graph_iri}> {{
    ?block a ex:Block ;
           ex:text ?text .
    FILTER(
      CONTAINS(LCASE(?text), "dataset") ||
      CONTAINS(LCASE(?text), "datasets") ||
      CONTAINS(LCASE(?text), "benchmark") ||
      CONTAINS(LCASE(?text), "corpus") ||
      CONTAINS(LCASE(?text), "data")
    )
  }}
}}
LIMIT 15
""".strip()

    if question_type == "metric_query":
        return f"""
PREFIX ex: <http://example.org/schema/>

SELECT ?block ?text
WHERE {{
  GRAPH <{graph_iri}> {{
    ?block a ex:Block ;
           ex:text ?text .
    FILTER(
      CONTAINS(LCASE(?text), "accuracy") ||
      CONTAINS(LCASE(?text), "f1") ||
      CONTAINS(LCASE(?text), "precision") ||
      CONTAINS(LCASE(?text), "recall") ||
      CONTAINS(LCASE(?text), "bleu") ||
      CONTAINS(LCASE(?text), "rouge") ||
      CONTAINS(LCASE(?text), "dice") ||
      CONTAINS(LCASE(?text), "iou") ||
      CONTAINS(LCASE(?text), "metric")
    )
  }}
}}
LIMIT 15
""".strip()

    if question_type == "figure_query":
        return f"""
PREFIX ex: <http://example.org/schema/>

SELECT ?figure ?captionText
WHERE {{
  GRAPH <{graph_iri}> {{
    ?figure a ex:Figure ;
            ex:hasCaptionBlock ?cap .
    ?cap ex:text ?captionText .
  }}
}}
LIMIT 15
""".strip()

    if question_type == "table_query":
        return f"""
PREFIX ex: <http://example.org/schema/>

SELECT ?table ?captionText
WHERE {{
  GRAPH <{graph_iri}> {{
    ?table a ex:Table ;
           ex:hasCaptionBlock ?cap .
    ?cap ex:text ?captionText .
  }}
}}
LIMIT 15
""".strip()

    if question_type == "result_query":
        return f"""
PREFIX ex: <http://example.org/schema/>

SELECT ?block ?text
WHERE {{
  GRAPH <{graph_iri}> {{
    ?block a ex:Block ;
           ex:text ?text .
    FILTER(
      CONTAINS(LCASE(?text), "result") ||
      CONTAINS(LCASE(?text), "results") ||
      CONTAINS(LCASE(?text), "performance") ||
      CONTAINS(LCASE(?text), "outperform") ||
      CONTAINS(LCASE(?text), "achieve") ||
      CONTAINS(LCASE(?text), "improve")
    )
  }}
}}
LIMIT 15
""".strip()

    return f"""
PREFIX ex: <http://example.org/schema/>

SELECT ?block ?text
WHERE {{
  GRAPH <{graph_iri}> {{
    ?block a ex:Block ;
           ex:blockType "caption" ;
           ex:text ?text .
  }}
}}
LIMIT 15
""".strip()


def shorten_text(text: str, max_len: int = 220) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def build_evidence_preview(question_type: str, bindings: List[dict], max_items: int = 5) -> List[dict]:
    preview = []

    for item in bindings[:max_items]:
        if question_type in {"dataset_query", "metric_query", "result_query", "general_caption_query"}:
            preview.append(
                {
                    "text": shorten_text(item.get("text", {}).get("value", "")),
                    "block": item.get("block", {}).get("value"),
                }
            )
        elif question_type == "figure_query":
            preview.append(
                {
                    "caption": shorten_text(item.get("captionText", {}).get("value", "")),
                    "figure": item.get("figure", {}).get("value"),
                }
            )
        elif question_type == "table_query":
            preview.append(
                {
                    "caption": shorten_text(item.get("captionText", {}).get("value", "")),
                    "table": item.get("table", {}).get("value"),
                }
            )

    return preview


def synthesize_answer(question_type: str, bindings: List[dict]) -> str:
    if not bindings:
        return "No strongly relevant graph evidence was found for this question."

    if question_type in {"dataset_query", "metric_query", "result_query", "general_caption_query"}:
        texts = []
        for b in bindings[:5]:
            if "text" in b:
                txt = b["text"]["value"]
                if txt not in texts:
                    texts.append(txt)

        if texts:
            return "Top supporting evidence:\n\n" + "\n\n".join(shorten_text(t, 300) for t in texts[:3])

    if question_type == "figure_query":
        captions = []
        for b in bindings[:5]:
            if "captionText" in b:
                txt = b["captionText"]["value"]
                if txt not in captions:
                    captions.append(txt)

        if captions:
            return "Relevant figure captions:\n\n" + "\n\n".join(shorten_text(t, 300) for t in captions[:3])

    if question_type == "table_query":
        captions = []
        for b in bindings[:5]:
            if "captionText" in b:
                txt = b["captionText"]["value"]
                if txt not in captions:
                    captions.append(txt)

        if captions:
            return "Relevant table captions:\n\n" + "\n\n".join(shorten_text(t, 300) for t in captions[:3])

    return "Graph query executed successfully."


def run_query_orchestrator_agent(
    *,
    question: str,
    doc_id: str,
    graphdb_base_url: str,
    repo_id: str,
) -> Dict:
    question_type = classify_question(question)
    sparql = generate_sparql(question_type, doc_id)

    try:
        result = sparql_query_graphdb(
            graphdb_base_url=graphdb_base_url,
            repo_id=repo_id,
            sparql=sparql,
        )
    except ConnectError:
        return {
            "agent_name": "query_orchestrator_agent",
            "status": "error",
            "question": question,
            "question_type": question_type,
            "generated_sparql": sparql,
            "num_results": 0,
            "answer": "GraphDB is not reachable. Start GraphDB and ensure the document has been ingested.",
            "evidence_preview": [],
            "raw_evidence": [],
            "error": {
                "type": "graphdb_connection_error",
                "graphdb_base_url": graphdb_base_url,
                "repo_id": repo_id,
            },
        }

    bindings = result.get("results", {}).get("bindings", [])
    answer = synthesize_answer(question_type, bindings)
    evidence_preview = build_evidence_preview(question_type, bindings, max_items=5)

    return {
        "agent_name": "query_orchestrator_agent",
        "status": "completed",
        "question": question,
        "question_type": question_type,
        "generated_sparql": sparql,
        "num_results": len(bindings),
        "answer": answer,
        "evidence_preview": evidence_preview,
        "raw_evidence": bindings[:10],
    }