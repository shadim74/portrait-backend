"""Speech-to-text with faster-whisper.

Accepts whatever the browser recorded (WebM/Opus, WAV, MP3...). faster-whisper
decodes it with PyAV, so no ffmpeg step is needed.
"""
import io

from .config import settings


class SpeechToText:
    def __init__(self):
        from faster_whisper import WhisperModel  # imported lazily: heavy

        self.model = WhisperModel(
            settings.whisper_model,
            device=settings.device,
            compute_type=settings.compute_type,
        )

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        segments, _info = self.model.transcribe(
            io.BytesIO(audio_bytes),
            language=language,
            beam_size=1,          # greedy: fastest, accuracy loss is tiny
            vad_filter=True,      # drops silence, reduces hallucinated text
        )
        return " ".join(s.text.strip() for s in segments).strip()
