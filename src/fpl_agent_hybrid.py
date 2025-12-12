import time
from typing import Dict, Any, List, Optional

# --- IMPORTS ---
# Ensure these files are in the same directory
from config import TOP_K
from llm_utils import get_llm_instance
from fpl_agent_baseline import parse_user_intent, run_cypher
from fpl_agent_embeddings import perform_semantic_search

# [cite_start]--- PROMPT TEMPLATE [cite: 71, 72, 73, 74, 75] ---
# Designed to meet M3 requirements: Context, Persona, Task
HYBRID_PROMPT_TEMPLATE = """
--- PERSONA ---
You are an expert FPL (Fantasy Premier League) Assistant. 
Your goal is to help users optimize their fantasy teams using data-driven insights.

--- CONTEXT ---
Use the following retrieved information to answer the user's question.
The information comes from two sources:
1. Database Records (Exact stats from the graph database)
2. Semantic Search (Textual descriptions of player profiles)

{context_str}

--- TASK ---
Answer the user's question: "{user_query}"
1. Base your answer PRIMARILY on the "Database Records" if they exist, as they are most accurate.
2. Use "Semantic Search" to add context or if specific stats are missing.
3. If the information is not in the context, admit you don't know. Do not hallucinate stats.
4. Keep the answer concise and helpful for an FPL manager.

Answer:
"""

def format_context(cypher_results: List[Dict], vector_results: List[Dict]) -> str:
    """
    Merges Structured (Cypher) and Unstructured (Vector) results into a single context string.
    [cite_start][cite: 68, 69]
    """
    context_parts = []

    # 1. Format Structured Data (Baseline)
    if cypher_results:
        context_parts.append("### Database Records (High Confidence):")
        for record in cypher_results:
            # Flatten dict to string (e.g., "Player: Haaland, Goals: 36")
            record_str = ", ".join([f"{k}: {v}" for k, v in record.items()])
            context_parts.append(f"- {record_str}")
    else:
        context_parts.append("### Database Records: No direct match found.")

    context_parts.append("\n")

    # 2. Format Unstructured Data (Embeddings)
    if vector_results:
        context_parts.append("### Semantic Search (Contextual):")
        for res in vector_results:
            # Use 'text' key from the vector store result
            text = res.get('text', '')
            context_parts.append(f"- {text}")
    else:
        context_parts.append("### Semantic Search: No relevant profiles found.")

    return "\n".join(context_parts)

def process_query(
    user_query: str, 
    llm_key: str = "gemma", 
    emb_key: str = "minilm", 
    use_cypher: bool = True, 
    use_vector: bool = True
) -> Dict[str, Any]:
    """
    Main Hybrid RAG Pipeline.
    
    Args:
        user_query: The user's question.
        llm_key: "gemma", "llama", or "gemini" (Selects the LLM).
        emb_key: "minilm" or "bge" (Selects the embedding model).
        use_cypher: Enable/Disable Baseline retrieval.
        use_vector: Enable/Disable Vector retrieval.
        
    Returns:
        Dict with keys: 'answer', 'context_used', 'logs', 'duration', 'model_used'
    """
    
    start_time = time.time()
    
    # [cite_start]Initialize logs for UI transparency [cite: 97, 98]
    logs = {
        "intent": None, 
        "cypher_params": None, 
        "retrieved_cypher": [], 
        "retrieved_vector": []
    }
    
    # [cite_start]--- STEP 1: INTENT CLASSIFICATION (Preprocessing) [cite: 10, 11] ---
    # We parse intent even if Cypher is disabled, as it helps understand the query structure.
    intent_data = parse_user_intent(user_query)
    intent = intent_data.get("intent")
    params = intent_data.get("parameters")
    
    logs["intent"] = intent
    logs["cypher_params"] = params
    
    # [cite_start]--- STEP 2: RETRIEVAL LAYER [cite: 27, 28] ---
    
    # [cite_start]A. Baseline Retrieval (Cypher) [cite: 29]
    cypher_results = []
    if use_cypher:
        if intent and intent != "error":
            try:
                cypher_results = run_cypher(intent, params)
                logs["retrieved_cypher"] = cypher_results
            except Exception as e:
                print(f"[Hybrid Error] Cypher failed: {e}")
        else:
            print("[Hybrid] Cypher skipped (No valid intent detected)")

    # [cite_start]B. Embedding Retrieval (Vector Search) [cite: 51]
    vector_results = []
    if use_vector:
        try:
            # Pass the selected embedding key (emb_key) to the search function
            vector_results = perform_semantic_search(user_query, model_key=emb_key, k=TOP_K)
            logs["retrieved_vector"] = vector_results
        except Exception as e:
            print(f"[Hybrid Error] Vector search failed: {e}")

    # [cite_start]--- STEP 3: CONTEXT MERGING [cite: 65, 68] ---
    context_str = format_context(cypher_results, vector_results)
    
    # [cite_start]--- STEP 4: LLM GENERATION [cite: 64, 71] ---
    # Dynamically load the requested model
    llm = get_llm_instance(llm_key)
    
    if not llm:
        return {
            "answer": "Error: Could not initialize LLM. Check API keys or Model Name.", 
            "logs": logs,
            "context_used": context_str
        }

    # Construct the final grounded prompt
    final_prompt = HYBRID_PROMPT_TEMPLATE.format(
        context_str=context_str,
        user_query=user_query
    )
    
    try:
        # Invoke LLM
        response = llm.invoke(final_prompt)
        # Normalize response (handle different LangChain return types)
        answer = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        answer = f"Error during LLM generation: {e}"

    duration = round(time.time() - start_time, 2)

    return {
        "answer": answer,
        "context_used": context_str,
        "logs": logs,
        "duration": duration,
        "model_used": llm_key
    }

# --- CLI FOR INTERACTIVE TESTING ---
if __name__ == "__main__":
    print("--- FPL HYBRID AGENT (CLI) ---")
    print("Configure your session:")
    
    # 1. Select Model
    print("Available Models: [gemma, llama, gemini]")
    sel_llm = input("Choose LLM (default: gemma): ").strip() or "gemma"
    
    # 2. Select Embedding Model
    print("Available Embeddings: [minilm, bge]")
    sel_emb = input("Choose Embedding (default: minilm): ").strip() or "minilm"
    
    # 3. Toggle Retrieval
    sel_cypher = input("Enable Cypher? (y/n, default: y): ").strip().lower() != 'n'
    sel_vector = input("Enable Vector? (y/n, default: y): ").strip().lower() != 'n'

    print(f"\nConfiguration: LLM={sel_llm} | Emb={sel_emb} | Cypher={sel_cypher} | Vector={sel_vector}")
    print("-" * 50)

    while True:
        q = input("\nAsk a question (or 'q' to quit): ")
        if q.lower() == 'q': break
        
        result = process_query(
            q, 
            llm_key=sel_llm, 
            emb_key=sel_emb, 
            use_cypher=sel_cypher, 
            use_vector=sel_vector
        )
        
        print("\n" + "="*20 + " RESULTS " + "="*20)
        print(f"Intent Detected: {result['logs']['intent']}")
        print(f"Cypher Records: {len(result['logs']['retrieved_cypher'])}")
        print(f"Vector Chunks: {len(result['logs']['retrieved_vector'])}")
        print("-" * 50)
        print(f"ANSWER ({result['duration']}s):")
        print(result['answer'])
        print("="*50)