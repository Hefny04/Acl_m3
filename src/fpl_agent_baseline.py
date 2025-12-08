import os
import json
import re
from typing import Any, List, Optional

# --- IMPORTS ---
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models.llms import LLM
from huggingface_hub import InferenceClient
from neo4j import GraphDatabase

# 1. SETUP NEO4J CONNECTION
URI = "neo4j+s://1cc4cb6e.databases.neo4j.io"
AUTH = ("neo4j", "gJLmWmdcogS5Fh1G1OuQaqIzz6PiehlnoXv8RCSxx6s")

# 2. SETUP CUSTOM LLM WRAPPER
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

class FreeHFChatLLM(LLM):
    repo_id: str = "google/gemma-2-2b-it"
    api_token: str = os.environ["HUGGINGFACEHUB_API_TOKEN"]
    client: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = InferenceClient(model=self.repo_id, token=self.api_token)

    @property
    def _llm_type(self) -> str:
        return "custom_hf_chat"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            # Temperature 0 for maximum determinism
            response = self.client.chat_completion(messages, max_tokens=500, temperature=0.0)
            return response.choices[0].message.content
        except Exception as e:
            return f"API_ERROR: {e}"

llm = FreeHFChatLLM()

# --- SECTION A: ROBUST CYPHER TEMPLATES ---
# Adapted specifically for Create_kg (1).py naming conventions
CYPHER_TEMPLATES = {
    # 1. Player Summary
    "player_summary": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        RETURN p.player_name AS Player,
               sum(r.total_points) AS TotalPoints,
               sum(r.goals_scored) AS Goals,
               sum(r.assists) AS Assists,
               sum(r.minutes) AS Minutes
    """,

    # 2. Top Players
    "top_players_by_position": """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
        WHERE toLower(pos.name) = toLower($position) OR toLower(pos.name) = toLower($position_mapped)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, sum(coalesce(r.total_points, 0)) AS TotalPoints
        ORDER BY TotalPoints DESC
        LIMIT toInteger($limit)
        RETURN p.player_name AS Player, TotalPoints
    """,

    # 3. Player vs Team
    "player_vs_team": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team)
        WHERE toLower(t.name) CONTAINS toLower($opponent)
        RETURN f.fixture_number AS GW,
               t.name AS Opponent,
               r.total_points AS Points,
               r.goals_scored AS Goals,
               r.assists AS Assists
    """,

    # 4. Squad by Position
    # USES HEURISTIC: Appears in >2 games involving Team X = Plays for Team X
    "team_squad_by_position": """
        MATCH (t:Team) WHERE toLower(t.name) CONTAINS toLower($team)
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
        WHERE toLower(pos.name) = toLower($position) OR toLower(pos.name) = toLower($position_mapped)
        
        MATCH (p)-[:PLAYED_IN]->(f:Fixture {season: $season})
        MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)
        WITH p, count(f) as games
        WHERE games > 2
        RETURN DISTINCT p.player_name AS Player
        LIMIT toInteger($limit)
    """,

    # 5. Compare Players
    "compare_players": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE any(name IN $player_names WHERE toLower(p.player_name) CONTAINS toLower(name))
        RETURN p.player_name AS Player,
               sum(r.total_points) AS TotalPoints,
               sum(r.goals_scored) AS Goals,
               sum(r.assists) AS Assists
    """,

    # 6. Team Performance in GW
    "team_performance_in_gw": """
        MATCH (s:Season {season_name: $season})-[:HAS_GW]->(g:Gameweek {GW_number: toInteger($gw)})
        MATCH (g)-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (t:Team)<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f)
        WHERE toLower(t.name) CONTAINS toLower($team)
        MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(opponent:Team)
        WHERE opponent.name <> t.name
        
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        // Heuristic: Filter for players who play for T (stats > 0 helps filter bench)
        MATCH (p)-[:PLAYED_IN]->(f_all:Fixture {season: $season})
        MATCH (f_all)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)
        WITH g, t, opponent, p, r, count(f_all) as squad_games
        WHERE squad_games > 2
        
        WITH g, t, opponent, sum(r.goals_scored) as TeamGoals, sum(r.total_points) as TeamPoints, collect(p.player_name)[0..3] as KeyPlayers
        RETURN g.GW_number AS GW, opponent.name AS Opponent, TeamGoals, TeamPoints, KeyPlayers
    """,

    # 7. Differentials
    "recommend_differentials": """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
        WHERE toLower(pos.name) = toLower($position) OR toLower(pos.name) = toLower($position_mapped)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, avg(r.influence) as AvgInf, sum(r.total_points) as Points
        WHERE Points > 30 
        RETURN p.player_name AS Player, toInteger(AvgInf) as Influence, Points
        ORDER BY AvgInf DESC
        LIMIT toInteger($limit)
    """,

    # 8. Best Captains
    "best_captain_options": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, r ORDER BY f.fixture_number DESC
        WITH p, collect(r.total_points)[0..3] as last_3_games
        WITH p, reduce(s = 0, x IN last_3_games | s + x) as form_points
        ORDER BY form_points DESC
        LIMIT 5
        RETURN p.player_name AS Player, form_points AS FormLast3GWs
    """,

    # 9. Availability
    "player_availability_check": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        WITH p, r, f ORDER BY f.fixture_number DESC LIMIT 3
        RETURN p.player_name, collect(r.minutes) as Last3Minutes
    """,

    # 10. Highest Score
    "highest_scoring_gw": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        MATCH (s:Season)-[:HAS_GW]->(g:Gameweek)-[:HAS_FIXTURE]->(f)
        RETURN g.GW_number AS GW, r.total_points AS Points
        ORDER BY Points DESC
        LIMIT 1
    """
}

# --- SECTION B: ROBUST MAPPINGS & CLEANING ---

def clean_json_string(json_str):
    """
    Fixes common JSON errors from small LLMs before parsing.
    Specifically removes trailing commas in objects.
    """
    # Remove code block markers if present
    json_str = json_str.replace("```json", "").replace("```", "").strip()
    # Regex to remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
    return json_str

def normalize_params(params):
    if not params: params = {}
    
    # 1. Parameter Key Normalization
    mappings = {
        "player": "player_name",
        "name": "player_name",
        "home_team": "team",
        "away_team": "opponent"
    }
    cleaned = {}
    for k, v in params.items():
        new_key = mappings.get(k, k)
        cleaned[new_key] = v

    # 2. Type Enforcement (Critical for Neo4j)
    if "limit" in cleaned:
        try: cleaned["limit"] = int(cleaned["limit"])
        except: cleaned["limit"] = 5
    else:
        cleaned["limit"] = 5
        
    if "gw" in cleaned:
        try: cleaned["gw"] = int(cleaned["gw"])
        except: pass

    # 3. Position Mapping (Case Insensitive)
    pos_map = {
        "goalkeeper": "GK", "gk": "GK",
        "defender": "DEF", "def": "DEF", "defence": "DEF",
        "midfielder": "MID", "mid": "MID", "midfield": "MID",
        "forward": "FWD", "striker": "FWD", "fwd": "FWD", "attack": "FWD"
    }
    
    if "position" in cleaned:
        raw_pos = str(cleaned["position"]).lower()
        cleaned["position_mapped"] = pos_map.get(raw_pos, "MID")
    else:
        cleaned["position"] = "MID"
        cleaned["position_mapped"] = "MID"

    # 4. Team Name Mapping (Fuzzy handling)
    # The KG has "Man City", user might ask "Manchester City"
    team_map = {
        "manchester city": "Man City",
        "manchester united": "Man Utd",
        "man utd": "Man Utd",
        "nottingham": "Nott'm Forest",
        "tottenham": "Spurs",
        "wolves": "Wolves",
        "sheffield": "Sheffield Utd",
        "luton": "Luton",
        "newcastle": "Newcastle"
    }
    
    for key in ["team", "opponent"]:
        if key in cleaned:
            val = cleaned[key].lower()
            for long_name, short_name in team_map.items():
                if long_name in val:
                    cleaned[key] = short_name
                    break

    # 5. Season Logic (CRITICAL FIX)
    # Your "Old Baseline" logs show success with 2022-23. The new one defaulted to 2023-24.
    # We switch the default back to 2022-23 to match your data.
    if "season" not in cleaned or cleaned["season"] in ["current", "this season", "last season"]:
        cleaned["season"] = "2022-23" 
    
    # If user explicitly types 2023-24, we keep it, but default is 22-23.
    
    return cleaned

def parse_user_intent(user_query):
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="""
        You are an expert FPL assistant. Map the user's question to the correct intent and parameters.
        
        INTENTS:
        1. "player_summary": Stats for 1 player (points, goals, minutes). 
           - Trigger: "perform", "stats for X", "summary". Params: player_name, season.
        2. "top_players_by_position": Best/Top players ranking. 
           - Params: position, season, limit.
        3. "player_vs_team": Player performance against a SPECIFIC OPPONENT. 
           - Trigger: "against [Team]", "vs [Team]". Params: player_name, opponent, season.
        4. "team_squad_by_position": List players belonging to a team. 
           - Trigger: "List [Position]s for [Team]", "Who are [Team] [Position]s". Params: team, position.
        5. "compare_players": Compare 2 or more PLAYERS. 
           - Trigger: "X vs Y", "Compare X and Y". Params: player_names (list).
        6. "team_performance_in_gw": How a team did in a specific gameweek. 
           - Trigger: "How did [Team] do in GW X", "Did [Team] score in GW X". Params: team, gw.
        7. "recommend_differentials": Asking for "differentials", "scout", "underrated". 
           - Params: position.
        8. "best_captain_options": "Captain", "Form". 
           - Params: season.
        9. "player_availability_check": "Playing recently?", "Minutes", "Available?". 
           - Params: player_name.
        10. "highest_scoring_gw": "Best gameweek", "Highest score". 
           - Params: player_name.

        CRITICAL CORRECTION RULES:
        - "How did [Player] perform?" -> ALWAYS "player_summary".
        - "[Player A] vs [Player B]" -> ALWAYS "compare_players".
        - "Show me [Team] [Position]s" -> ALWAYS "team_squad_by_position".
        
        RETURN PURE JSON ONLY. NO MARKDOWN. NO TRAILING COMMAS.
        Example: {{ "intent": "player_summary", "parameters": {{ "player_name": "Haaland", "season": "2022-23" }} }}
        
        User Query: {query}
        """,
        input_variables=["query"],
    )

    chain = prompt | llm 
    try:
        # Get raw string first to clean it
        raw_response = chain.invoke({"query": user_query})
        cleaned_json = clean_json_string(raw_response)
        return json.loads(cleaned_json)
    except Exception as e:
        print(f"[LLM Error] Parsing Failed. Raw: {raw_response} | Error: {e}")
        return {"intent": "error", "parameters": {}}

# --- EXECUTION ENGINE ---
def execute_query(intent_data):
    intent = intent_data.get("intent")
    if intent: intent = intent.strip().lower()
    
    raw_params = intent_data.get("parameters")
    params = normalize_params(raw_params)
    
    print(f"[DEBUG] Intent: '{intent}' | Normalized Params: {params}")

    if intent not in CYPHER_TEMPLATES:
        return f"Error: Intent '{intent}' is not in the template library."
    
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            result = driver.execute_query(CYPHER_TEMPLATES[intent], params)
            records = result.records
            
            if not records: 
                return "No data found. (Check if Season/Names are correct or if data exists for this year)"
            
            output = []
            for r in records:
                output.append(json.dumps(r.data()))
            return "\n".join(output)
            
    except Exception as e:
        return f"Query Execution Error: {e}"

if __name__ == "__main__":
    print("--- FPL Graph-RAG Baseline Assistant (Robust V3) ---")
    
    while True:
        q = input("\nAsk a question (or 'q' to quit): ")
        if q.lower() == 'q': break
        try:
            structured_data = parse_user_intent(q)
            answer = execute_query(structured_data)
            print(f"\nGraph Response:\n{answer}")
        except Exception as e:
            print(f"System Error: {e}")



            #shsh#