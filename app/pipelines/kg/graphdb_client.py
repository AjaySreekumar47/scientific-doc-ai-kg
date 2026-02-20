from __future__ import annotations

import httpx


def graphdb_statements_url(graphdb_base_url: str, repo_id: str) -> str:
    return f"{graphdb_base_url.rstrip('/')}/repositories/{repo_id}/statements"


def graphdb_query_url(graphdb_base_url: str, repo_id: str) -> str:
    return f"{graphdb_base_url.rstrip('/')}/repositories/{repo_id}"


def upload_ttl_to_graphdb(
    graphdb_base_url: str,
    repo_id: str,
    ttl: str,
    context_iri: str | None = None,
    timeout_s: float = 60.0,
) -> int:
    """
    Uploads Turtle to GraphDB repository statements endpoint.
    Returns HTTP status code (204 is typical success).
    """
    url = graphdb_statements_url(graphdb_base_url, repo_id)

    params = {}
    if context_iri:
        params["context"] = f"<{context_iri}>"

    headers = {"Content-Type": "text/turtle"}

    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, params=params, content=ttl.encode("utf-8"), headers=headers)
        r.raise_for_status()
        return r.status_code


def sparql_query_graphdb(
    graphdb_base_url: str,
    repo_id: str,
    sparql: str,
    timeout_s: float = 60.0,
) -> dict:
    """
    Runs a SPARQL query against GraphDB and returns JSON results.
    """
    url = graphdb_query_url(graphdb_base_url, repo_id)
    headers = {"Accept": "application/sparql-results+json"}
    data = {"query": sparql}

    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, data=data, headers=headers)
        r.raise_for_status()
        return r.json()