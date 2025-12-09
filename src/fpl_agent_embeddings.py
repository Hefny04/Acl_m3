"""Embedding-augmented FPL assistant.

This script performs semantic search over player profile embeddings stored in
Neo4j and asks a lightweight Hugging Face chat model to answer using only the
retrieved context.
"""

import os
import difflib
from typing import Any, List, Optional

from huggingface_hub import InferenceClient
from langchain_community.vectorstores import Neo4jVector
from langchain_core.language_models.llms import LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

HF_TOKEN = "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"
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_NAME = "player_profile_index"


class FreeHFChatLLM(LLM):
    """Simple wrapper around a free Hugging Face Inference chat endpoint."""

    repo_id: str = "google/gemma-2-2b-it"
    api_token: str = os.environ["HUGGINGFACEHUB_API_TOKEN"]
    client: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = InferenceClient(model=self.repo_id, token=self.api_token)

    @property
    def _llm_type(self) -> str:  # pragma: no cover - required by LangChain
        return "custom_hf_chat"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, max_tokens=500, temperature=0.0)
        return response.choices[0].message.content


def load_vector_store() -> Neo4jVector:
    """Return an existing Neo4j vector store built by `create_embeddings.py`."""

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name=INDEX_NAME,
        text_node_property="text",
        embedding_node_property="embedding",
    )
    return vector_store


def rerank_by_player_name(question: str, docs: List) -> List:
    """Prioritize documents whose player name best matches the question."""

    def score(doc) -> float:
        name = doc.metadata.get("player_name", "").lower()
        if not name:
            return 0.0
        # SequenceMatcher gives a 0-1 score; higher is better.
        return difflib.SequenceMatcher(a=question.lower(), b=name).ratio()

    return sorted(docs, key=score, reverse=True)


def semantic_search(question: str, k: int = 5):
    """Return ranked embedding matches with both text and metadata."""

    store = load_vector_store()
    docs = store.similarity_search(question, k=k)
    ranked = rerank_by_player_name(question, docs)
    return [
        {"text": doc.page_content, "metadata": doc.metadata}
        for doc in ranked
    ]


def build_prompt(question: str, docs) -> str:
    if not docs:
        return (
            "No relevant player profiles were found for that question. "
            "Try rephrasing with the exact player name or regenerate embeddings for the season you want."
        )

    context_lines = []
    for doc in docs:
        context_lines.append(
            f"Player: {doc['metadata'].get('player_name', 'Unknown')}. Profile: {doc['text']}"
        )
    context = "\n".join(context_lines)
    template = PromptTemplate(
        template=(
            "You are an FPL expert. Use only the provided player profiles to answer the user.\n"
            "Context:\n{context}\n\n"
            "User Question: {question}\n"
            "Answer concisely. If the context is insufficient, state that explicitly."
        ),
        input_variables=["context", "question"],
    )
    chain = template | RunnableLambda(lambda data: llm.invoke(template.format(**data)))
    return chain.invoke({"context": context, "question": question})


llm = FreeHFChatLLM()


def main():
    print("--- FPL Embedding-Based Assistant ---")
    while True:
        user_q = input("\nAsk an FPL question (or 'q' to quit): ")
        if user_q.lower() == "q":
            break
        docs = semantic_search(user_q, k=5)
        response = build_prompt(user_q, docs)
        print(f"\nAnswer:\n{response}\n")


if __name__ == "__main__":
    main()
