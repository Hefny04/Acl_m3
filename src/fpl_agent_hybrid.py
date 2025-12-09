"""Hybrid Graph + Embedding FPL assistant.

This module combines the Cypher-based intent handling from `fpl_agent_baseline`
with the embedding search flow from `fpl_agent_embeddings` to produce grounded
answers that mix structured statistics and profile snippets.
"""

from collections import defaultdict
from typing import Any, Dict, List

from fpl_agent_baseline import FreeHFChatLLM, parse_user_intent, run_cypher
from fpl_agent_embeddings import semantic_search

llm = FreeHFChatLLM()


def _player_key(record: Dict[str, Any]) -> str:
    """Best-effort player identifier for merging records."""

    for key in ("Player", "player_name", "player"):
        if key in record and record[key]:
            return str(record[key]).lower()
    return ""


def _format_structured(record: Dict[str, Any]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in record.items())


def merge_context(cypher_data: List[Dict[str, Any]], semantic_data: List[Dict[str, Any]]):
    grouped: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"stats": [], "profiles": []})

    for record in cypher_data:
        key = _player_key(record)
        grouped[key]["stats"].append(_format_structured(record))

    for item in semantic_data:
        meta = item.get("metadata", {})
        key = str(meta.get("player_name", "")).lower()
        snippet = item.get("text", "")
        grouped[key]["profiles"].append(snippet)

    context_lines = []
    for key, payload in grouped.items():
        header = f"Player: {key or 'unknown'}"
        stats = " | ".join(payload["stats"]) if payload["stats"] else "No structured stats"
        profiles = " | ".join(payload["profiles"]) if payload["profiles"] else "No profile snippets"
        context_lines.append(f"{header}\nStats: {stats}\nProfile: {profiles}")

    return "\n\n".join(context_lines)


def build_hybrid_answer(question: str) -> str:
    intent_data = parse_user_intent(question)
    cypher_results = run_cypher(intent_data.get("intent"), intent_data.get("parameters"))
    embedding_results = semantic_search(question, k=5)

    context_block = merge_context(cypher_results, embedding_results)

    prompt = f"""
Context:
{context_block}

Persona:
You are an expert Fantasy Premier League data assistant. Use both the structured
stats and semantic snippets above. If the information is insufficient, say so
clearly.

Task:
Answer the user's question accurately and concisely using only the provided
context. Avoid fabricating stats. User question: {question}
"""

    return llm(prompt)


def main():
    print("--- FPL Hybrid Assistant (Graph + Embeddings) ---")
    while True:
        user_q = input("\nAsk an FPL question (or 'q' to quit): ")
        if user_q.lower() == "q":
            break
        try:
            answer = build_hybrid_answer(user_q)
            print(f"\nAnswer:\n{answer}\n")
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
