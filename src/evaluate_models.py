import time
import pandas as pd
from fpl_agent_hybrid import process_query

# 1. Define your "Golden Set" of Test Cases [cite: 83]
# These should cover different intents (Stats, Comparison, Listing, Reasoning)
TEST_CASES = [
    "How did Haaland perform in the 2022-23 season?",       # Intent: player_summary
    "Compare Saka and Martinelli.",                         # Intent: compare_players
    "Who were the top 5 midfielders?",                      # Intent: top_players_by_position
    "List the defenders for Arsenal.",                      # Intent: team_squad_by_position
    "Suggest some differential forwards for my team.",      # Intent: recommend_differentials (Reasoning)
]

# 2. Define Models to Test [cite: 77]
# Ensure these keys match what is in your llm_utils.py
MODELS = ["gemma", "llama", "gemini"] 

def run_evaluation():
    results = []
    print(f" Starting Evaluation on {len(MODELS)} models with {len(TEST_CASES)} test cases...\n")

    for model in MODELS:
        print(f"--- Testing Model: {model.upper()} ---")
        
        for query in TEST_CASES:
            print(f" Q: {query}")
            
            # A. Quantitative Metric: Response Time 
            start_time = time.time()
            
            try:
                # Force Hybrid Mode (Cypher + Vector) for a fair test
                response_data = process_query(
                    query, 
                    llm_key=model, 
                    emb_key="minilm",  # Keep embedding constant to isolate LLM performance
                    use_cypher=True, 
                    use_vector=True
                )
                answer = response_data["answer"]
                error = "None"
            except Exception as e:
                answer = "ERROR"
                error = str(e)
                
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            
            # Store Result
            results.append({
                "Model": model,
                "Question": query,
                "Response Time (s)": duration,
                "Answer Length (chars)": len(answer),
                "Final Answer": answer,
                "Error": error,
                # Placeholders for Manual Grading 
                "Qualitative: Accuracy (1-5)": "", 
                "Qualitative: Naturalness (1-5)": ""
            })
            print(f"Done in {duration}s")

    # 3. Save to CSV for analysis
    df = pd.DataFrame(results)
    output_file = "model_evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nEvaluation Complete! Results saved to '{output_file}'.")
    print(" Open this CSV in Excel/Sheets to perform your Qualitative Grading.")

if __name__ == "__main__":
    run_evaluation()