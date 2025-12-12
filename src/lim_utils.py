import os
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from huggingface_hub import InferenceClient
from config import HF_TOKEN, GEMINI_API_KEY

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

# --- CONFIGURATION ---
LLM_CONFIGS = {
    "gemma": {
        "type": "hf", 
        "repo_id": "google/gemma-2-2b-it", 
        "description": "Gemma 2B (Fast)"
    },
    "llama": {
        "type": "hf", 
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct", 
        "description": "Llama 3 8B (Smart)"
    },
    "gemini": {
        "type": "google", 
        "repo_id": "gemini-2.5-pro", 
        "description": "Gemini 2.5 pro"
    },
        "gemini_flash": {
        "type": "google", 
        "repo_id": "gemini-2.5-flash", 
        "description": "Gemini 2.5 Flash"
    },
}

class FreeHFChatLLM(LLM):
    """Wrapper for Hugging Face Inference API."""
    repo_id: str
    api_token: str = os.environ.get("HUGGINGFACEHUB_API_TOKEN", HF_TOKEN)
    client: Any = None

    def __init__(self, repo_id: str, **kwargs):
        super().__init__(repo_id=repo_id, **kwargs)
        self.client = InferenceClient(model=self.repo_id, token=self.api_token)

    @property
    def _llm_type(self) -> str:
        return "custom_hf_chat"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        try:
            # Gemma/Llama expect chat-formatted messages
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(
                messages, 
                max_tokens=500, 
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from Hugging Face API: {e}"

def get_llm_instance(llm_key):
    """Factory function to return the requested LLM object."""
    if llm_key not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM key: {llm_key}")
        
    config = LLM_CONFIGS[llm_key]
    
    if config["type"] == "google":
        if not ChatGoogleGenerativeAI:
            return None # Handle missing library gracefully in UI
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
        return ChatGoogleGenerativeAI(model=config["repo_id"], temperature=0.0)
    
    else: # Hugging Face
        return FreeHFChatLLM(repo_id=config["repo_id"])