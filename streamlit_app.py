from __future__ import annotations

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"


st.set_page_config(page_title="Autonomous Scientific KG Agent", layout="wide")
st.title("Autonomous Scientific Knowledge Graph Agent")

st.markdown(
    """
Upload a scientific PDF, run the agent pipeline, inspect the execution trace,
and query the resulting knowledge graph with provenance-backed answers.
"""
)

if "last_doc_id" not in st.session_state:
    st.session_state.last_doc_id = None

tabs = st.tabs(["Ingest", "Agent Trace", "Ask Graph", "Evidence Viewer"])


# ------------------------
# TAB 1 — INGEST
# ------------------------
with tabs[0]:
    st.subheader("Ingest PDF into GraphDB")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="ingest_pdf")
    user_goal = st.text_input(
        "Optional user goal",
        value="build graph for question answering",
        key="ingest_goal",
    )

    if st.button("Run Ingestion", key="run_ingestion"):
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        else:
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
            }
            params = {"user_goal": user_goal}

            with st.spinner("Running ingestion pipeline..."):
                resp = requests.post(
                    f"{API_BASE}/ingest_to_graphdb",
                    files=files,
                    params=params,
                    timeout=300,
                )

            if resp.ok:
                data = resp.json()
                st.session_state.last_doc_id = data["document_id"]

                st.success("Ingestion complete.")
                st.json(data)
            else:
                st.error(f"Ingestion failed: {resp.status_code}")
                st.text(resp.text)


# ------------------------
# TAB 2 — AGENT TRACE
# ------------------------
with tabs[1]:
    st.subheader("Agent Trace")

    uploaded_trace_file = st.file_uploader("Upload PDF for trace", type=["pdf"], key="trace_pdf")
    trace_goal = st.text_input(
        "Optional planning goal",
        value="extract methods and results",
        key="trace_goal",
    )

    if st.button("Run Agent Trace", key="run_trace"):
        if uploaded_trace_file is None:
            st.warning("Please upload a PDF first.")
        else:
            files = {
                "file": (uploaded_trace_file.name, uploaded_trace_file.getvalue(), "application/pdf")
            }
            params = {"user_goal": trace_goal}

            with st.spinner("Running agent trace..."):
                resp = requests.post(
                    f"{API_BASE}/agent_trace",
                    files=files,
                    params=params,
                    timeout=300,
                )

            if resp.ok:
                data = resp.json()
                st.success("Agent trace complete.")
                st.json(data)
            else:
                st.error(f"Agent trace failed: {resp.status_code}")
                st.text(resp.text)


# ------------------------
# TAB 3 — ASK GRAPH
# ------------------------
with tabs[2]:
    st.subheader("Ask the Graph")

    doc_id = st.text_input(
        "Document ID",
        value=st.session_state.last_doc_id or "",
        key="ask_doc_id",
    )
    question = st.text_area(
        "Question",
        value="What is the main result?",
        key="ask_question",
    )

    if st.button("Ask Graph", key="run_ask"):
        if not doc_id.strip():
            st.warning("Please provide a document ID.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            payload = {
                "doc_id": doc_id.strip(),
                "question": question.strip(),
            }

            with st.spinner("Querying graph..."):
                resp = requests.post(
                    f"{API_BASE}/ask_graph",
                    json=payload,
                    timeout=300,
                )

            if resp.ok:
                data = resp.json()
                st.success("Query complete.")

                st.markdown("### Answer")
                st.write(data.get("answer", ""))

                st.markdown("### Question Type")
                st.code(data.get("question_type", ""))

                st.markdown("### Generated SPARQL")
                st.code(data.get("generated_sparql", ""), language="sparql")

                st.markdown("### Evidence Preview")
                st.json(data.get("evidence_preview", []))

                with st.expander("Raw Evidence"):
                    st.json(data.get("raw_evidence", []))
            else:
                st.error(f"Ask-graph failed: {resp.status_code}")
                st.text(resp.text)


# ------------------------
# TAB 4 — EVIDENCE VIEWER
# ------------------------
with tabs[3]:
    st.subheader("Evidence Viewer")

    ev_doc_id = st.text_input(
        "Document ID for evidence",
        value=st.session_state.last_doc_id or "",
        key="ev_doc_id",
    )
    figure_id = st.text_input("Figure ID", key="ev_figure_id")
    table_id = st.text_input("Table ID", key="ev_table_id")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load Figure", key="load_figure"):
            if not ev_doc_id or not figure_id:
                st.warning("Provide both document ID and figure ID.")
            else:
                fig_url = f"{API_BASE}/docs/{ev_doc_id}/figures/{figure_id}/image"
                st.image(fig_url, caption=f"Figure: {figure_id}")

    with col2:
        if st.button("Load Table", key="load_table"):
            if not ev_doc_id or not table_id:
                st.warning("Provide both document ID and table ID.")
            else:
                tbl_url = f"{API_BASE}/docs/{ev_doc_id}/tables/{table_id}/image"
                st.image(tbl_url, caption=f"Table: {table_id}")