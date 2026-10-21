"""Portrait characters and per-connection conversation history."""
import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings

SPEECH_RULES = (
    "\n\nYou are speaking out loud through a talking portrait. Reply in 1 to 3 short "
    "spoken sentences. Talk like a real person today: plain, casual, modern English, "
    "contractions, everyday words. Never use old-fashioned words like sir, madam, "
    "indeed, alas, pray, thee or 'I must say'. Ask the visitor things back now and then. "
    "No lists, no markdown, no emoji, no stage directions. "
    "Stay in character. /no_think"  # /no_think turns off Qwen3's slow reasoning mode
)


@dataclass
class Persona:
    id: str
    name: str
    system_prompt: str
    voice: str = "af_heart"
    portrait: str = ""
    greeting: str = ""
    welcome_back: str = ""   # said instead of the greeting to a returning visitor
    # Optional, for PNG/JPG portraits: where the mouth is, in % of the image,
    # e.g. {"x": 44, "y": 55, "w": 12, "h": 9}. The page moves that area as a jaw.
    mouth: dict | None = None
    # Optional 3D model shown instead of the image, e.g.
    # {"file": "portraits/models/butler/butler.fbx", "textures": {"head": {"color": "..."}}}
    model: dict | None = None


def load_personas(path: str | None = None) -> dict[str, Persona]:
    data = json.loads(Path(path or settings.personas_file).read_text(encoding="utf-8"))
    return {p["id"]: Persona(**p) for p in data}


@dataclass
class Conversation:
    persona: Persona
    history: list[dict] = field(default_factory=list)

    def messages_for(self, user_text: str) -> list[dict]:
        self.history.append({"role": "user", "content": user_text})
        self.history = self.history[-settings.max_history:]
        system = {"role": "system", "content": self.persona.system_prompt + SPEECH_RULES}
        return [system, *self.history]

    def add_reply(self, text: str):
        if text:
            self.history.append({"role": "assistant", "content": text})
