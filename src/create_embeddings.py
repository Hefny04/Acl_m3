import os
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
URI = "neo4j+s://1cc4cb6e.databases.neo4j.io"
AUTH = ("neo4j", "gJLmWmdcogS5Fh1G1OuQaqIzz6PiehlnoXv8RCSxx6s")

# Using a standard, free embedding model from Hugging Face
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def create_player_embeddings():
    print("--- 1. Connecting to Neo4j ---")
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    # Query to fetch aggregated player data for the text chunk
    # We construct a natural language string for the embedding
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
    
    print("--- 2. Fetching Player Data & Constructing Text ---")
    data = []
    with driver.session() as session:
        result = session.run(fetch_query)
        for record in result:
            data.append({"id": record["id"], "text": record["text"]})
    
    driver.close()
    print(f"Prepared {len(data)} player profiles for embedding.")

    # --- 3. Create Vector Index using LangChain ---
    print(f"--- 3. Generating Embeddings ({EMBEDDING_MODEL}) & Indexing ---")
    
    # This automatically:
    # 1. Embeds the 'text' using the HF model
    # 2. Stores it in a new node property 'embedding'
    # 3. Creates a Vector Index named 'player_profile_index'
    
    try:
        vector_store = Neo4jVector.from_texts(
            texts=[d["text"] for d in data],
            metadatas=[{"player_name": d["id"]} for d in data], # Metadata helps retrieval
            embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            url=URI,
            username=AUTH[0],
            password=AUTH[1],
            index_name="player_profile_index",
            node_label="PlayerProfile",  # We create specific nodes for vector search
            text_node_property="text",
            embedding_node_property="embedding"
        )
        print("Success! Vector Index 'player_profile_index' created.")
    except Exception as e:
        print(f"Error creating vector index: {e}")

if __name__ == "__main__":
    create_player_embeddings()