import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --- IMPORTS ---
from fpl_agent_hybrid import process_query
from llm_utils import LLM_CONFIGS 
from fpl_agent_embeddings import EMBEDDING_CONFIGS

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FPL Chat Assistant",
    page_icon="⚽",
    layout="wide"
)

# --- HELPER: GRAPH VISUALIZATION ---
def render_graph(raw_struct):
    """Generates a NetworkX graph from Cypher results."""
    try:
        G = nx.Graph()
        # Limit nodes to avoid clutter
        for i, rec in enumerate(raw_struct[:10]):
            center_node = rec.get('player') or rec.get('Player') or rec.get('Team') or f"Record {i}"
            G.add_node(center_node, color='red', size=20)
            
            for k, v in rec.items():
                if k not in ['player', 'Player', 'Team', 'team']:
                    attr_node = f"{k}: {v}"
                    G.add_node(attr_node, color='skyblue', size=10)
                    G.add_edge(center_node, attr_node)
        
        if len(G.nodes) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                    node_size=500, font_size=8, ax=ax)
            st.pyplot(fig)
            plt.close(fig) # Clean up memory
    except Exception as e:
        st.warning(f"Could not visualize graph: {e}")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Retrieval Strategy")
    use_cypher = st.checkbox("Baseline (Structured)", value=True)
    use_vector = st.checkbox("Semantic (Vector)", value=True)
    
    st.divider()

    # Logic for Model Selection
    is_hybrid = use_cypher and use_vector
    
    if is_hybrid:
        llm_choice = st.selectbox("LLM Model:", options=list(LLM_CONFIGS.keys()), index=0)
    else:
        llm_choice = "gemma"
        st.caption("🔒 Model locked to 'Gemma' for single-source queries.")

    if use_vector:
        emb_choice = st.selectbox("Embedding Model:", options=list(EMBEDDING_CONFIGS.keys()), index=0)
    else:
        emb_choice = "minilm" 
        
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("Fantasy Premiere League AI Assistant")
st.markdown("Ask me about players, stats, fixtures, or transfer targets.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Display Chat History
# [START OF CHAT INTERFACE SECTION]

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If it's an assistant message, check for stored details (graphs/tables)
        if message["role"] == "assistant" and "details" in message:
            details = message["details"]
            logs = details.get("logs", {})
            duration = details.get("duration", 0.0)
            
            # Display Time
            st.caption(f"Answered in **{duration}s** using {details.get('model_used', 'unknown')}")

            # Expandable "White Box" view for this specific message
            with st.expander("View Retrieval Details"):
                
                # A. Structured Data
                st.markdown("### 1. Structured Data (Cypher)")
                
                # --- NEW: Display the Triggered Intent ---
                intent = logs.get("intent", "None")
                st.markdown(f"**Triggered Template:** `{intent}`") 
                # -----------------------------------------

                raw_struct = logs.get("retrieved_cypher", [])
                if raw_struct:
                    st.dataframe(pd.DataFrame(raw_struct))
                 #   render_graph(raw_struct)
                else:
                    st.info("No structured data found.")

                st.divider()

                # B. Unstructured Data
                st.markdown("### 2. Semantic Context (Vector)")
                semantic_docs = logs.get("retrieved_vector", [])
                if semantic_docs:
                    for i, doc in enumerate(semantic_docs):
                        with st.popover(f"Source Chunk {i+1}"):
                            st.write(doc.get('text', ''))
                            st.json(doc.get('metadata', {}))
                else:
                    st.info("No semantic context found.")

# 2. Chat Input
if prompt := st.chat_input("Ex: Who are the best differential defenders?"):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing FPL Data..."):
            try:
                # Call Backend
                results = process_query(
                    prompt, 
                    llm_key=llm_choice, 
                    emb_key=emb_choice, 
                    use_cypher=use_cypher, 
                    use_vector=use_vector
                )
                
                # Display Answer
                st.markdown(results["answer"])
                st.caption(f"Answered in **{results['duration']}s**")
                
                # Append to history with full details object
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": results["answer"],
                    "details": results 
                })
                
                # Force rerun to render the expander/details properly in the loop above
                st.rerun()
                
            except Exception as e:
                st.error(f"An error occurred: {e}")