"""FastAPI service: POST /ask returns the agent's answer with its sources.

Run:  DOCS_DIR=docs ANTHROPIC_API_KEY=... uvicorn rag_agent.api:app --reload
Without an API key the service runs in "search only" mode and returns the best passages.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import DocsAgent
from .llm import AnthropicLLM
from .retrieval import Retriever, load_documents


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Answer(BaseModel):
    answer: str
    sources: list[str]
    steps: list[str] = []
    mode: str


def create_app(docs_dir: Path | None = None, llm=None) -> FastAPI:
    docs_dir = docs_dir or Path(os.environ.get("DOCS_DIR", "docs"))
    retriever = Retriever(load_documents(docs_dir))
    if llm is None and os.environ.get("ANTHROPIC_API_KEY"):
        llm = AnthropicLLM()
    agent = DocsAgent(llm, retriever) if llm else None

    app = FastAPI(title="RAG docs agent")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(retriever.chunks), "mode": "agent" if agent else "search"}

    @app.post("/ask", response_model=Answer)
    def ask(q: Question) -> Answer:
        if agent:
            r = agent.ask(q.question)
            return Answer(answer=r.answer, sources=r.sources, steps=r.steps, mode="agent")
        hits = retriever.search(q.question, k=3)
        if not hits:
            raise HTTPException(404, "No relevant passage found")
        return Answer(answer=hits[0].chunk.text,
                      sources=[f"{h.chunk.source}#{h.chunk.index}" for h in hits], mode="search")

    return app


app = create_app() if Path(os.environ.get("DOCS_DIR", "docs")).exists() else None
