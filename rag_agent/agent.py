"""A small tool-calling agent that answers questions from a document base, with citations."""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import LLM
from .retrieval import Retriever

SYSTEM_PROMPT = (
    "You answer questions using ONLY the company documents returned by the search_docs tool. "
    "Always search before answering. Cite sources as [file.md#chunk]. "
    "If the documents do not contain the answer, say so instead of guessing. "
    "Use the calculator tool for any arithmetic."
)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def safe_calc(expression: str) -> float:
    """Evaluate a basic arithmetic expression without eval()."""
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression: {expression!r}")
    return _eval(ast.parse(expression, mode="eval").body)


@dataclass
class AgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)


class DocsAgent:
    def __init__(self, llm: LLM, retriever: Retriever, max_steps: int = 5):
        self.llm, self.retriever, self.max_steps = llm, retriever, max_steps
        self.tools: dict[str, tuple[dict, Callable[..., Any]]] = {
            "search_docs": ({
                "name": "search_docs",
                "description": "Search the internal documents. Returns the most relevant passages.",
                "input_schema": {"type": "object", "properties": {
                    "query": {"type": "string"}, "k": {"type": "integer", "default": 3}},
                    "required": ["query"]},
            }, self._search),
            "calculator": ({
                "name": "calculator",
                "description": "Evaluate an arithmetic expression such as '12 * (3 + 4)'.",
                "input_schema": {"type": "object", "properties": {
                    "expression": {"type": "string"}}, "required": ["expression"]},
            }, lambda expression: str(safe_calc(expression))),
        }
        self._sources: list[str] = []

    def _search(self, query: str, k: int = 3) -> str:
        hits = self.retriever.search(query, k=k)
        if not hits:
            return "NO_RESULTS"
        out = []
        for h in hits:
            ref = f"{h.chunk.source}#{h.chunk.index}"
            if ref not in self._sources:
                self._sources.append(ref)
            out.append(f"[{ref}] (score {h.score:.2f}) {h.chunk.text}")
        return "\n".join(out)

    def ask(self, question: str) -> AgentResult:
        self._sources = []
        messages: list[dict] = [{"role": "user", "content": question}]
        schemas = [schema for schema, _ in self.tools.values()]
        steps: list[str] = []
        for _ in range(self.max_steps):
            reply = self.llm.complete(SYSTEM_PROMPT, messages, schemas)
            if not reply.tool_calls:
                return AgentResult(reply.text.strip(), list(self._sources), steps)
            assistant_content: list[dict] = []
            if reply.text:
                assistant_content.append({"type": "text", "text": reply.text})
            results: list[dict] = []
            for call in reply.tool_calls:
                assistant_content.append({"type": "tool_use", "id": call.id,
                                          "name": call.name, "input": call.arguments})
                steps.append(f"{call.name}({call.arguments})")
                if call.name not in self.tools:
                    output, is_error = f"unknown tool {call.name}", True
                else:
                    try:
                        output, is_error = self.tools[call.name][1](**call.arguments), False
                    except Exception as exc:  # tool errors are returned to the model, not raised
                        output, is_error = f"error: {exc}", True
                results.append({"type": "tool_result", "tool_use_id": call.id,
                                "content": output, "is_error": is_error})
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": results})
        return AgentResult("Stopped: too many steps without a final answer.", list(self._sources), steps)
