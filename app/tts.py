"""Text-to-speech with Kokoro-82M (Apache 2.0). Runs on GPU or CPU."""
import io
import wave

import numpy as np

from .visemes import timeline

SAMPLE_RATE = 24_000  # Kokoro's native output rate


def to_wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """float32 [-1, 1] mono samples -> 16-bit PCM WAV file bytes."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class TextToSpeech:
    def __init__(self):
        from kokoro import KPipeline  # imported lazily: heavy

        self._pipelines: dict[str, "KPipeline"] = {}
        self._KPipeline = KPipeline

    def _pipeline(self, voice: str):
        # Kokoro voices are prefixed by language: a=US English, b=UK English, ...
        lang = voice[0]
        if lang not in self._pipelines:
            self._pipelines[lang] = self._KPipeline(lang_code=lang)
        return self._pipelines[lang]

    def synthesize(self, text: str, voice: str = "af_heart", speed: float = 1.0) -> bytes:
        return self.synthesize_timed(text, voice, speed)[0]

    def synthesize_timed(self, text: str, voice: str = "af_heart", speed: float = 1.0) -> tuple[bytes, list]:
        """WAV bytes plus a lip-sync timeline [[seconds, viseme], ...]."""
        chunks, visemes, offset = [], [], 0.0
        for result in self._pipeline(voice)(text, voice=voice, speed=speed):
            audio = result.audio
            if audio is None:
                continue
            samples = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            if result.pred_dur is not None:
                visemes += timeline(result.phonemes, result.pred_dur.tolist(), offset)
            offset += len(samples) / SAMPLE_RATE
            chunks.append(samples)
        samples = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        return to_wav_bytes(samples.astype(np.float32)), visemes
