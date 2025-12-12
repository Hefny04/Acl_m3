import os
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, HF_TOKEN

# --- CONFIGURATION ---
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", HF_TOKEN)
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

EMBEDDING_CONFIGS = {
    "bge": {
        "model_name": "BAAI/bge-small-en-v1.5", 
        "index_name": "player_profile_index_bge", 
        "embedding_property": "embedding_bge"
    },
    "minilm": {
        "model_name": "all-MiniLM-L6-v2", 
        "index_name": "player_profile_index_minilm", 
        "embedding_property": "embedding_minilm"
    }
}

def get_vector_store(model_key):
    """Initializes connection to specific Neo4j Vector Index."""
    conf = EMBEDDING_CONFIGS[model_key]
    return Neo4jVector.from_existing_index(
        embedding=HuggingFaceEmbeddings(model_name=conf["model_name"]),
        url=NEO4J_URI, 
        username=NEO4J_USER, 
        password=NEO4J_PASSWORD,
        index_name=conf["index_name"],
        text_node_property="text", 
        embedding_node_property=conf["embedding_property"]
    )

def rerank_by_player_name(question, docs):
    """Heuristic: Boosts documents that explicitly contain the player name mentioned in the query."""
    question_lower = question.lower()
    boosted = []
    unboosted = []
    
    for doc in docs:
        name = doc.metadata.get("player_name", "").lower()
        # Simple check: is the player name (or last name) in the question?
        if name and (name in question_lower or name.split()[-1] in question_lower):
            boosted.append(doc)
        else:
            unboosted.append(doc)
            
    return boosted + unboosted

def perform_semantic_search(question, model_key="minilm", k=5):
    """
    Main API function.
    Returns a clean list of dicts: [{'text': '...', 'metadata': {...}}, ...]
    """
    try:
        store = get_vector_store(model_key)
        docs = store.similarity_search(question, k=k)
        ranked_docs = rerank_by_player_name(question, docs)
        
        # Convert to simple list of dicts for the Hybrid Agent / UI
        return [
            {"text": d.page_content, "metadata": d.metadata} 
            for d in ranked_docs
        ]
    except Exception as e:
        print(f"[Vector Search Error] {e}")
        return []

if __name__ == "__main__":
    # ONLY used for testing this file in isolation
    from llm_utils import get_llm_instance
    from langchain_core.prompts import PromptTemplate

    print("--- FPL EMBEDDING AGENT (CLI TEST) ---")
    model_choice = input("Choose embedding model (bge/minilm): ").strip()
    if model_choice not in EMBEDDING_CONFIGS: model_choice = "minilm"
    
    llm = get_llm_instance("gemma") # Default for CLI test

    while True:
        q = input(f"\nAsk Vector Agent ({model_choice}) (q to quit): ")
        if q.lower() == 'q': break
        
        print("... Retrieving ...")
        results = perform_semantic_search(q, model_choice)
        
        print(f"Found {len(results)} context chunks.")
        
        # Simple Generation just for CLI proof-of-concept
        context_str = "\n".join([r['text'] for r in results])
        template = PromptTemplate(template="Context: {context}\n\nQ: {q}\nAnswer:", input_variables=["context", "q"])
        chain = template | llm
        print(f"Answer: {chain.invoke({'context': context_str, 'q': q})}")