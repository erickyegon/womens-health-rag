"""
Streamlit UI — Episode 26
===========================
Full chat interface with streaming, source citations panel, routing badge.
"""
from __future__ import annotations
import json, os, time
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY  = os.getenv("API_KEY", "dev-secret-change-in-prod")
HEADERS  = {"Authorization": f"Bearer {API_KEY}"}


def main():
    st.set_page_config(
        page_title="Women's Health Intelligence Assistant",
        page_icon="🌍",
        layout="wide",
    )

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Settings")
        use_agent   = st.toggle("Use LangGraph Agent", value=False,
                                help="Phase 3 agentic retrieval")
        show_sources = st.toggle("Show source citations", value=True)
        country_filter = st.selectbox("Filter by country",
                                      ["All", "Nigeria", "Kenya", "Ghana", "Ethiopia"])
        year_filter = st.selectbox("Filter by year",
                                   ["All", "2019", "2021", "2022"])
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Women's Health RAG — Production Course")
        st.caption("Built with LangChain + LangGraph + pgvector")

        # Health check
        try:
            r = requests.get(f"{API_BASE}/health", timeout=2)
            st.success(f"API: {r.json()['status']}")
        except Exception:
            st.error("API unreachable")

    # ── Main UI ────────────────────────────────────────────────────────────────
    st.title("🌍 Women's Health Intelligence Assistant")
    st.caption("Grounded in DHS reports from Nigeria, Kenya, Ghana, and Ethiopia")

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") and show_sources:
                _render_sources(msg["sources"])

    # Chat input
    if question := st.chat_input("Ask a question about women's health data..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder  = st.empty()
            sources_slot = st.empty()
            meta_slot    = st.empty()

            # Build filters
            filters = {}
            if country_filter != "All":
                filters["country"] = country_filter
            if year_filter != "All":
                filters["year"] = year_filter

            # Stream response
            full_text = ""
            sources   = []
            t0        = time.perf_counter()

            try:
                with requests.post(
                    f"{API_BASE}/query/stream",
                    headers=HEADERS,
                    json={"question": question, "filters": filters,
                          "use_agent": use_agent},
                    stream=True, timeout=60,
                ) as resp:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "token" in data:
                                    full_text += data["token"]
                                    placeholder.markdown(full_text + "▌")
                                if "error" in data:
                                    placeholder.error(data["error"])
                            except Exception:
                                pass

            except requests.exceptions.ConnectionError:
                full_text = ("⚠️ Cannot reach the API. "
                             "Make sure `make up` is running.")

            placeholder.markdown(full_text)
            elapsed = time.perf_counter() - t0

            if show_sources and sources:
                with sources_slot.expander(f"📄 Sources ({len(sources)})"):
                    _render_sources(sources)

            meta_slot.caption(f"⏱ {elapsed:.1f}s | "
                              f"{'🤖 Agent' if use_agent else '⚡ Chain'}")

        st.session_state.messages.append({
            "role": "assistant", "content": full_text, "sources": sources})


def _render_sources(sources: list[dict]):
    for s in sources:
        st.markdown(
            f"**[Source {s.get('n','')}]** "
            f"{s.get('title','Unknown')} | "
            f"{s.get('country','')} {s.get('year','')} | "
            f"Page {s.get('page','?')}"
        )


if __name__ == "__main__":
    main()
