"""Utility to generate player profile embeddings for Neo4j vector search.

This script fetches aggregated season stats for each player, builds a concise
natural-language profile string, embeds it with a Hugging Face model, and stores
it in a Neo4j vector index for similarity search.
"""

import os
from typing import List, Tuple

from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

# --- CONFIGURATION ---
# Default Hugging Face token supplied by the user; environment override allowed.
HF_TOKEN = "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"
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Free, high-quality embedding model available on Hugging Face.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_NAME = "player_profile_index"
NODE_LABEL = "PlayerProfile"


def fetch_player_profiles(driver) -> List[Tuple[str, str]]:
    """Return (id, profile_text) pairs describing each player."""

    fetch_query = """
    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: '2022-23'})
    MATCH (p)-[:PLAYS_AS]->(pos:Position)

    // Aggregate stats per player for the season
    WITH p, pos,
         sum(r.total_points) as total_points,
         sum(r.goals_scored) as goals,
         sum(r.assists) as assists,
         avg(r.influence) as influence,
         count(f) as appearances

    // Construct the text to embed
    WITH p,
         "Player: " + p.player_name +
         ". Position: " + pos.name +
         ". Season 2022-23 Stats: " +
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
    """Create/update the Neo4j vector index populated with player profiles."""

    print("--- 1. Connecting to Neo4j ---")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print("--- 2. Fetching Player Data & Constructing Text ---")
    player_profiles = fetch_player_profiles(driver)
    print(f"Prepared {len(player_profiles)} player profiles for embedding.")

    print(f"--- 3. Generating Embeddings ({EMBEDDING_MODEL}) & Indexing ---")
    try:
        Neo4jVector.from_texts(
            texts=[profile for _, profile in player_profiles],
            metadatas=[{"player_name": player} for player, _ in player_profiles],
            embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name=INDEX_NAME,
            node_label=NODE_LABEL,
            text_node_property="text",
            embedding_node_property="embedding",
        )
        print(f"Success! Vector Index '{INDEX_NAME}' created or updated.")
    finally:
        driver.close()


if __name__ == "__main__":
    create_player_embeddings()
