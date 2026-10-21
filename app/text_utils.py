"""Pure text helpers for streaming: no model dependencies, easy to test."""
import re

# A sentence ends at . ! ? … followed by whitespace (or a closing quote/bracket
# then whitespace). We only cut once whitespace arrives, so "3.5" or "Dr." at the
# very end of a chunk is never split prematurely.
_SENTENCE_END = re.compile(r'([.!?…]+["\')\]]*)\s+')
_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "e.g.", "i.e.", "etc."}
# A clause ends at , ; : (then whitespace) or at a dash: a natural spot to pause.
_CLAUSE_END = re.compile(r'[,;:]["\')\]]*\s+|\s*[—–]\s*')
MIN_CHARS = 12  # avoid sending tiny fragments like "Ah." alone to TTS


class SentenceSplitter:
    """Feed streamed tokens in, get speakable chunks out.

    Chunks are normally whole sentences. To start talking sooner, the first
    chunk of a reply is cut at the first clause break after `first_min` chars,
    and any sentence that grows past `long_len` chars is cut at a clause break
    after `long_min` chars.
    """

    def __init__(self, first_min: int = 24, long_len: int = 110, long_min: int = 40):
        self.buffer = ""
        self.emitted = 0
        self.first_min, self.long_len, self.long_min = first_min, long_len, long_min

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        chunks = []
        start = 0
        for m in _SENTENCE_END.finditer(self.buffer):
            candidate = self.buffer[start:m.end()].strip()
            last_word = candidate.split()[-1].lower() if candidate else ""
            if last_word in _ABBREVIATIONS or len(candidate) < MIN_CHARS:
                continue  # keep accumulating
            chunks.append(candidate)
            start = m.end()
        self.buffer = self.buffer[start:]
        self.emitted += len(chunks)
        chunks += self._clauses()
        return chunks

    def _clauses(self) -> list[str]:
        out = []
        while True:
            if self.emitted == 0:
                min_pos = self.first_min
            elif len(self.buffer) >= self.long_len:
                min_pos = self.long_min
            else:
                break
            m = next((m for m in _CLAUSE_END.finditer(self.buffer) if m.start() >= min_pos), None)
            if not m:
                break
            out.append(self.buffer[:m.end()].strip())
            self.buffer = self.buffer[m.end():]
            self.emitted += 1
        return out

    def flush(self) -> list[str]:
        rest = self.buffer.strip()
        self.buffer = ""
        return [rest] if rest else []


class ThinkFilter:
    """Removes <think>...</think> blocks from a token stream.

    Qwen3 and other reasoning models may emit hidden reasoning first; the
    portrait must never speak it aloud.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.inside = False
        self.pending = ""

    def feed(self, text: str) -> str:
        self.pending += text
        out = []
        while self.pending:
            tag = self.CLOSE if self.inside else self.OPEN
            idx = self.pending.find(tag)
            if idx >= 0:
                if not self.inside:
                    out.append(self.pending[:idx])
                self.pending = self.pending[idx + len(tag):]
                self.inside = not self.inside
                continue
            # No full tag. Hold back a possible partial tag at the end.
            keep = 0
            for k in range(1, len(tag)):
                if self.pending.endswith(tag[:k]):
                    keep = k
            emit, self.pending = self.pending[:len(self.pending) - keep], self.pending[len(self.pending) - keep:]
            if not self.inside:
                out.append(emit)
            break
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.pending
        self.pending = ""
        return rest


def clean_for_speech(text: str) -> str:
    """Strip markdown and emoji-ish symbols the TTS would read literally."""
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)
    return re.sub(r"\s+", " ", text).strip()
