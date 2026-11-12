from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_agent.agent import DocsAgent, safe_calc
from rag_agent.api import create_app
from rag_agent.llm import FakeLLM, LLMReply, ToolCall
from rag_agent.retrieval import Retriever, chunk_text, load_documents

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_chunking_respects_size_and_overlap():
    text = " ".join(f"Phrase numéro {i} sur un sujet donné." for i in range(60))
    chunks = chunk_text(text, "x.md", max_chars=200, overlap=40)
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[1].text[:20] in chunks[0].text  # overlap keeps context


def test_chunking_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", "x.md", max_chars=100, overlap=100)


def test_retriever_finds_the_right_document():
    r = Retriever(load_documents(DOCS, max_chars=300, overlap=60))
    assert r.search("combien de jours de télétravail par semaine ?")[0].chunk.source == "teletravail.md"
    assert r.search("plafond pour une nuit d'hôtel à Paris")[0].chunk.source == "notes_de_frais.md"
    assert r.search("RTT cadres")[0].chunk.source == "conges.md"


def test_safe_calc():
    assert safe_calc("2.70 * 22") == pytest.approx(59.4)
    with pytest.raises(ValueError):
        safe_calc("__import__('os').system('ls')")


def test_agent_uses_tools_and_cites_sources():
    llm = FakeLLM([
        LLMReply(tool_calls=[ToolCall("t1", "search_docs", {"query": "indemnité télétravail par jour"})]),
        LLMReply(tool_calls=[ToolCall("t2", "calculator", {"expression": "2.70 * 12"})]),
        LLMReply(text="12 jours donnent 32,40 euros [teletravail.md#0]."),
    ])
    agent = DocsAgent(llm, Retriever(load_documents(DOCS, max_chars=300, overlap=60)))
    result = agent.ask("Combien touche-t-on pour 12 jours de télétravail ?")
    assert "32,40" in result.answer
    assert any(s.startswith("teletravail.md") for s in result.sources)
    assert result.steps[0].startswith("search_docs") and result.steps[1].startswith("calculator")
    # the calculator result was sent back to the model as a tool_result
    last_turn = llm.calls[-1][-1]["content"][0]
    assert last_turn["type"] == "tool_result" and last_turn["content"].startswith("32.4")


def test_agent_reports_tool_errors_to_the_model():
    llm = FakeLLM([
        LLMReply(tool_calls=[ToolCall("t1", "calculator", {"expression": "open('x')"})]),
        LLMReply(text="Je ne peux pas calculer cela."),
    ])
    agent = DocsAgent(llm, Retriever(load_documents(DOCS)))
    agent.ask("test")
    assert llm.calls[-1][-1]["content"][0]["is_error"] is True


def test_agent_stops_after_max_steps():
    loop = [LLMReply(tool_calls=[ToolCall(f"t{i}", "search_docs", {"query": "congés"})]) for i in range(3)]
    agent = DocsAgent(FakeLLM(loop), Retriever(load_documents(DOCS)), max_steps=3)
    assert agent.ask("?").answer.startswith("Stopped")


def test_api_search_mode_and_agent_mode():
    client = TestClient(create_app(DOCS))
    assert client.get("/health").json()["mode"] == "search"
    r = client.post("/ask", json={"question": "délai pour déposer une demande de congés"})
    assert r.status_code == 200 and r.json()["sources"][0].startswith("conges.md")

    llm = FakeLLM([LLMReply(tool_calls=[ToolCall("t1", "search_docs", {"query": "congés"})]),
                   LLMReply(text="15 jours avant [conges.md#0].")])
    client = TestClient(create_app(DOCS, llm=llm))
    body = client.post("/ask", json={"question": "Délai pour poser des congés ?"}).json()
    assert body["mode"] == "agent" and "15 jours" in body["answer"]
