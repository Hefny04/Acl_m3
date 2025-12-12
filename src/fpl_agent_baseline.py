import json
import re
from typing import Dict, Any, List
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DEFAULT_SEASON
from llm_utils import get_llm_instance
from cypher_template_2 import CYPHER_TEMPLATES

# --- 1. INTENT CLASSIFICATION (Updated for Scout/ICT Logic) ---
def parse_user_intent(query: str) -> Dict[str, Any]:
    """
    Uses an LLM to map natural language to a Cypher template key + parameters.
    """
    llm = get_llm_instance("gemini_flash") # Fast model for routing
    
    # Updated System Prompt to match your new "Scout-Heavy" templates
    system_instruction = """
    You are an intent classifier for an FPL AI Assistant.
    Map the user's query to one of the following INTENTS and extract parameters.
    
    --- AVAILABLE INTENTS ---
    1. "player_summary": General stats, points, goals for a player. 
       (e.g. "How did Haaland do?", "Stats for Salah")
    2. "top_players_by_position": Leaderboards, best players. 
       (e.g. "Best defenders", "Top scoring forwards")
    3. "player_vs_team": History against an opponent. 
       (e.g. "How does Kane perform against Arsenal?")
    4. "team_squad_by_position": List players from a specific team. 
       (e.g. "List Arsenal midfielders", "Defenders from Man City")
    5. "compare_players": Season-long comparison of 2+ players. 
       (e.g. "Compare Saka and Foden")
    6. "compare_players_last_5": Recent form comparison. 
       (e.g. "Who is in better form?", "Compare recent stats of Watkins and Isak")
    7. "recommend_differentials": Scouts players with high underlying stats (ICT). 
       (e.g. "Who has good underlying stats?", "Suggest a differential", "Hidden gems")
    8. "best_captain_options": Best captaincy picks based on form. 
       (e.g. "Who should I captain?", "Best captain")
    9. "highest_scoring_gw": Best single gameweek.
       (e.g. "What was Haaland's best gameweek?")

    --- OUTPUT FORMAT ---
    Return ONLY a JSON object:
    {
      "intent": "exact_key_from_above",
      "parameters": {
        "player_name": "...",
        "player_names": ["...", "..."], // For comparisons
        "team": "...",
        "position": "...", 
        "opponent": "...",
        "season": "2022-23" // Default if unspecified
      }
    }
    """
    
    prompt = f"{system_instruction}\n\nUser Query: {query}\nJSON:"
    
    try:
        response = llm.invoke(prompt)
        # Handle cases where LLM returns object vs string
        content = response.content if hasattr(response, 'content') else str(response)
        return clean_json_string(content)
    except Exception as e:
        print(f"Intent Parsing Error: {e}")
        return {"intent": "error", "parameters": {}}

def clean_json_string(json_str: str) -> Dict[str, Any]:
    """Helper to sanitize LLM JSON output."""
    try:
        # Remove markdown code blocks if present
        clean = re.sub(r"```json|```", "", json_str).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"intent": "error", "parameters": {}}

# --- 2. PARAMETER NORMALIZATION ---
def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure parameters match DB format.
    Handles key aliasing, type conversion, and entity mapping (Position/Teams).
    """
    if not params:
        params = {}

    # 1. Parameter Key Normalization
    key_mappings = {
        "player": "player_name",
        "name": "player_name",
        "home_team": "team",
        "away_team": "opponent"
    }
    
    cleaned_input = {}
    for k, v in params.items():
        new_key = key_mappings.get(k, k)
        cleaned_input[new_key] = v

    # 2. Define Defaults (Include 'position_mapped' to fix the crash)
    defaults = {
        "season": DEFAULT_SEASON,
        "limit": 5,
        "player_name": "",
        "opponent": "",
        "team": "",
        "player_names": [],
        "position": "",
        "position_mapped": ""  # <--- CRITICAL FIX: Ensure key exists
    }
    
    final_params = {**defaults, **cleaned_input}

    # 3. Type Enforcement
    try:
        final_params["limit"] = int(final_params["limit"])
    except (ValueError, TypeError):
        final_params["limit"] = 5
        
    if "gw" in final_params:
        try:
            final_params["gw"] = int(final_params["gw"])
        except (ValueError, TypeError):
            pass

    # 4. Position Mapping (Normalize to DB codes)
    pos_map = {
        "gkp": "GKP", "goalie": "GKP", "goalkeeper": "GKP", "gk": "GKP",
        "def": "DEF", "defender": "DEF", "defence": "DEF",
        "mid": "MID", "midfielder": "MID", "midfield": "MID",
        "fwd": "FWD", "forward": "FWD", "striker": "FWD", "attack": "FWD"
    }
    
    raw_pos = str(final_params.get("position", "")).lower()
    mapped_pos = pos_map.get(raw_pos)
    
    # If we found a valid mapping (e.g., "defender" -> "DEF")
    if mapped_pos:
        final_params["position"] = mapped_pos
        final_params["position_mapped"] = mapped_pos  # <--- FIX: Populate the specific key your query needs
    else:
        # If no position specified, keep defaults or raw value
        # Some queries might rely on position_mapped being non-empty? 
        # If your query is "WHERE p.position = $position_mapped", leaving it empty might return 0 results (which is fine) 
        # or error if the query doesn't handle empty strings.
        final_params["position_mapped"] = raw_pos if raw_pos else ""

    # 5. Team Name Mapping
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
        val = str(final_params.get(key, "")).lower()
        if val:
            for long_name, short_name in team_map.items():
                if long_name in val:
                    final_params[key] = short_name
                    break

    # 6. List Logic
    if not final_params["player_names"] and final_params["player_name"]:
        final_params["player_names"] = [final_params["player_name"]]

    return final_params

# --- 3. EXECUTION LAYER (Returns Raw Data now) ---
def run_cypher(intent: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Executes the Cypher query and returns RAW LIST OF DICTIONARIES.
    This integration is required for the Hybrid Agent.
    """
    if intent not in CYPHER_TEMPLATES:
        print(f"Warning: Intent '{intent}' not found in templates.")
        return []

    # Prepare logic
    query_template = CYPHER_TEMPLATES[intent]
    safe_params = normalize_params(params)
    
    results = []
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            result = session.run(query_template, safe_params)
            # Convert Neo4j records to standard Python dicts
            results = [dict(record) for record in result]
            limit = safe_params.get("limit", 5)
            if len(results) > limit:
                # print(f"[Info] Truncating results from {len(results)} to {limit}")
                results = results[:limit]
    except Exception as e:
        print(f"Cypher Execution Error: {e}")
    finally:
        driver.close()
        
    return results

if __name__ == "__main__":
    # Quick Test
    q = "Compare form of Salah and Saka"
    print(f"Testing Query: {q}")
    parsed = parse_user_intent(q)
    print(f"Parsed: {parsed}")
    if parsed["intent"] != "error":
        data = run_cypher(parsed["intent"], parsed["parameters"])
        print(f"Retrieved {len(data)} records.")
        print(data)