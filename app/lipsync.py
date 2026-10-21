"""Lip-sync: turns (portrait, speech audio) into a talking video clip.

Start with LIPSYNC_ENGINE=none: the server sends audio only and the frontend
plays an idle/talking loop over the portrait. Switch to musetalk once the rest
works end to end.
"""
from pathlib import Path

from .config import settings


class LipSync:
    """Base class. render() returns MP4 bytes, or None for audio-only mode."""

    def render(self, wav_bytes: bytes, portrait_path: str) -> bytes | None:
        return None


class MuseTalkLipSync(LipSync):
    """Adapter for MuseTalk 1.5 (https://github.com/TMElyralab/MuseTalk).

    Setup (one time, on a CUDA machine):
      git clone https://github.com/TMElyralab/MuseTalk third_party/MuseTalk
      cd third_party/MuseTalk && follow its README to install and download weights

    MuseTalk's realtime mode precomputes each portrait's face latents once
    ("preparation"), after which each new audio clip renders at ~30 fps. Wire
    that up here: load the models once in __init__, prepare each persona's
    portrait the first time it's seen, then in render() write wav_bytes to a
    temp file, run inference, and return the MP4 bytes.
    """

    def __init__(self):
        root = Path("third_party/MuseTalk")
        if not root.exists():
            raise RuntimeError(
                "LIPSYNC_ENGINE=musetalk but third_party/MuseTalk is missing. "
                "Clone and install MuseTalk first, or set LIPSYNC_ENGINE=none."
            )
        raise NotImplementedError(
            "MuseTalk adapter is a stub: implement model loading and render() "
            "following the docstring above."
        )


def make_lipsync() -> LipSync:
    if settings.lipsync_engine == "musetalk":
        return MuseTalkLipSync()
    return LipSync()
