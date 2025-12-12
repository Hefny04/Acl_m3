"""Utility to generate player profile embeddings for Neo4j vector search.

*** DEBUGGING VERSION ***
This version focuses ONLY on the failing 'minilm' model to diagnose the error.
"""

import os
from typing import List, Tuple

from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, HF_TOKEN

# --- CONFIGURATION ---
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", HF_TOKEN)
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# We are only testing the failing model configuration
MODEL_CONFIGS = [
    {
        "model_name": "BAAI/bge-small-en-v1.5",
        "index_name": "player_profile_index_bge",
        "embedding_property": "embedding_bge"  # <--- New unique property name
    },
    {
        "model_name": "all-MiniLM-L6-v2", 
        "index_name": "player_profile_index_minilm",
        "embedding_property": "embedding_minilm" # <--- New unique property name
    }
]
NODE_LABEL = "player_info" 


def fetch_player_profiles(driver) -> List[Tuple[str, str]]:
    """Return (id, profile_text) pairs describing each player."""

    # Using the same query as before
    fetch_query = """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        MATCH (p)-[:PLAYS_AS]->(pos:Position)
        WITH p, pos, f.season AS season,  // <--- 1. Group by season here so it's available
            sum(r.total_points) as total_points,
            sum(r.goals_scored) as goals,
            sum(r.assists) as assists,
            avg(r.influence) as influence,
            count(f) as appearances
        WITH p, 
            "Player: " + p.player_name +
            ". Position: " + pos.name +
            ". Season: " + season + " Stats: " +  // <--- 2. Concatenate the variable 'season'
            "Total Points: " + toString(total_points) +
            ", Goals: " + toString(goals) +
            ", Assists: " + toString(assists) +
            ", Appearances: " + toString(appearances) +
            ", Influence: " + toString(toInteger(influence)) AS text_description
        RETURN p.player_name AS id, text_description AS text
    """

    profiles: List[Tuple[str, str]] = []
    with driver.session() as session:
        for record in session.run(fetch_query):
            profiles.append((record["id"], record["text"]))
    return profiles


def create_player_embeddings():
    """Create the Neo4j vector index for the configured model, with explicit error logging."""

    print("--- 1. Connecting to Neo4j ---")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print("--- 2. Fetching Player Data & Constructing Text ---")
    player_profiles = fetch_player_profiles(driver)
    print(f"Prepared {len(player_profiles)} player profiles for embedding.")

    for config in MODEL_CONFIGS:
        model_name = config["model_name"]
        index_name = config["index_name"]
        
        print(f"\n--- 3. STARTING EMBEDDING GENERATION AND INDEXING for {model_name} ({index_name}) ---")

        try:
            # This is the line that generates vectors and creates the index
            Neo4jVector.from_texts(
                texts=[profile for _, profile in player_profiles],
                metadatas=[{"player_name": player} for player, _ in player_profiles],
                embedding=HuggingFaceEmbeddings(model_name=model_name),
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PASSWORD,
                index_name=index_name,
                node_label=NODE_LABEL,
                text_node_property="text",
                embedding_node_property=config["embedding_property"],
            )
            print(f"SUCCESS! Vector Index '{index_name}' created or updated.")
        except Exception as e:
            # IMPORTANT: Print the detailed error traceback to diagnose the issue
            import traceback
            print(f"FATAL ERROR: Index creation for {index_name} FAILED.")
            print("--- TRACEBACK ---")
            traceback.print_exc()
            print("-----------------")
        finally:
            pass
            
    driver.close()


if __name__ == "__main__":
    create_player_embeddings()