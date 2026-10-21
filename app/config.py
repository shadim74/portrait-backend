"""All settings come from environment variables, with sensible local defaults."""
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()  # reads .env in the working directory, if present
except ImportError:
    pass


def _default_compute_type(device: str) -> str:
    # float16 only works on GPU; int8 is the fast choice on CPU.
    return "float16" if device == "cuda" else "int8"


@dataclass(frozen=True)
class Settings:
    # LLM: any OpenAI-compatible server (Ollama, vLLM, LM Studio, llama.cpp server)
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3:8b")
    llm_api_key: str = os.getenv("LLM_API_KEY", "ollama")  # Ollama ignores it
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "300"))

    # Speech-to-text (faster-whisper)
    device: str = os.getenv("DEVICE", "cuda")  # "cuda" or "cpu"
    whisper_model: str = os.getenv("WHISPER_MODEL", "large-v3-turbo")
    compute_type: str = field(default="")

    # Lip-sync: "none" (audio only) or "musetalk"
    lipsync_engine: str = os.getenv("LIPSYNC_ENGINE", "none")

    personas_file: str = os.getenv("PERSONAS_FILE", "personas.json")
    max_history: int = int(os.getenv("MAX_HISTORY", "20"))  # messages the LLM sees per turn
    history_db: str = os.getenv("HISTORY_DB", "data/history.db")  # saved chats ("" = don't save)

    def __post_init__(self):
        ct = os.getenv("COMPUTE_TYPE") or _default_compute_type(self.device)
        object.__setattr__(self, "compute_type", ct)


settings = Settings()
