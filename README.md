# rag-docs-agent

A small **RAG + tool-calling agent** that answers questions about internal documents (HR policies, procedures, wikis) **with citations**, exposed as a **FastAPI** service.

It is the kind of assistant companies are asking for right now: "ask our documentation in plain language, get a sourced answer, and don't make things up".

## How it works

```
question ──► agent (LLM) ──► tool: search_docs ──► TF-IDF retriever over chunked .md/.txt files
                 │        └─► tool: calculator  ──► safe arithmetic (no eval)
                 └──► final answer + sources [file.md#chunk]
```

- **Ingestion & chunking**: documents are split into overlapping chunks on paragraph/sentence boundaries.
- **Retrieval**: character n-gram TF-IDF + cosine similarity (robust to plurals and typos, no external service). The `Retriever` class can be swapped for embeddings + a vector database.
- **Agent loop**: the LLM decides which tool to call; tool results (and tool errors) are sent back to the model; the loop is capped (`max_steps`) to avoid runaway calls.
- **Grounding**: the system prompt forces the model to search first, cite sources and say when the documents don't contain the answer.
- **LLM-agnostic**: one small `LLM` interface. An Anthropic client (native tool-use API) is included; a `FakeLLM` scripts replies for tests.
- **API**: `POST /ask` and `GET /health`. Without an API key the service still works in *search only* mode.

## Run

```bash
pip install -r requirements.txt
DOCS_DIR=docs uvicorn rag_agent.api:app --reload            # search-only mode
DOCS_DIR=docs ANTHROPIC_API_KEY=... uvicorn rag_agent.api:app  # agent mode
curl -X POST localhost:8000/ask -H "content-type: application/json" \
     -d '{"question": "Combien de jours de télétravail par semaine ?"}'
```

Or with Docker: `docker build -t rag-docs-agent . && docker run -p 8000:8000 rag-docs-agent`

The `docs/` folder contains three short **fictional** HR documents used for the demo and the tests.

## Tests

```bash
pytest -q
```
Covers chunking, retrieval quality, the agent loop (tool calls, tool errors, step limit, citations) and the API, using a scripted LLM so no network or API key is needed.

## Next steps

Embeddings + pgvector/Qdrant, re-ranking, evaluation set (answer accuracy / citation precision), streaming responses.

---
Simon Becart — software engineer (Python, API integration, AI applications) · [LinkedIn](https://www.linkedin.com/in/simonbecart/)
