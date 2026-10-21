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

    To start talking almost as soon as the text appears, the first chunks of a
    reply are tiny: each ends at the first pause (sentence end, comma, dash)
    after `first_min` chars, or simply after a few complete words
    (`early_words`: 4 words for the first chunk, 8 for the second) if no pause
    comes. The voice of chunk 1 plays while chunk 2 is generated, and so on.
    After that, chunks are whole sentences; a sentence that grows past
    `long_len` chars is cut at a clause break after `long_min` chars.
    """

    def __init__(self, first_min: int = 10, early_words: tuple = (4, 8), long_len: int = 110, long_min: int = 40):
        self.buffer = ""
        self.emitted = 0
        self.first_min, self.early_words = first_min, early_words
        self.long_len, self.long_min = long_len, long_min

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        chunks = []
        start = 0
        for m in _SENTENCE_END.finditer(self.buffer):
            candidate = self.buffer[start:m.end()].strip()
            last_word = candidate.split()[-1].lower() if candidate else ""
            min_chars = self.first_min if self.emitted + len(chunks) < len(self.early_words) else MIN_CHARS
            if last_word in _ABBREVIATIONS or len(candidate) < min_chars:
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
            early = self.emitted < len(self.early_words)
            if early:
                min_pos = self.first_min
            elif len(self.buffer) >= self.long_len:
                min_pos = self.long_min
            else:
                break
            m = next((m for m in _CLAUSE_END.finditer(self.buffer) if m.start() >= min_pos), None)
            if m:
                cut = m.end()
            elif early and (cut := self._after_words(self.early_words[self.emitted])):
                pass                                    # no pause yet: cut after N whole words
            else:
                break
            out.append(self.buffer[:cut].strip())
            self.buffer = self.buffer[cut:]
            self.emitted += 1
        return out

    def _after_words(self, n: int) -> int:
        """Index just after the n-th complete word (one followed by a space), or 0."""
        count = 0
        for m in re.finditer(r"\S+\s+", self.buffer):
            count += 1
            if count == n:
                return m.end()
        return 0

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


EMOTIONS = ("neutral", "happy", "sad", "surprised", "angry", "thinking")
_EMOTION_ALIASES = {"smile": "happy", "smiling": "happy", "amused": "happy", "laughing": "happy",
                    "excited": "happy", "joyful": "happy", "warm": "happy", "playful": "happy",
                    "unhappy": "sad", "wistful": "sad", "melancholy": "sad", "upset": "sad",
                    "shocked": "surprised", "surprise": "surprised", "curious": "thinking",
                    "thoughtful": "thinking", "pensive": "thinking", "annoyed": "angry",
                    "irritated": "angry", "mad": "angry", "calm": "neutral"}
GESTURES = ("laugh", "shrug", "nod", "shake", "wave", "think")
_GESTURE_ALIASES = {"laughs": "laugh", "laughing": "laugh", "chuckle": "laugh", "chuckles": "laugh",
                    "giggle": "laugh", "shrugs": "shrug", "shrugging": "shrug", "nods": "nod",
                    "nodding": "nod", "shake head": "shake", "shakes head": "shake",
                    "head shake": "shake", "shaking head": "shake", "waves": "wave", "waving": "wave",
                    "hand on chin": "think", "ponder": "think", "ponders": "think"}
_TAG = re.compile(r"\[\s*([A-Za-z][A-Za-z \-]{0,18}?)\s*\]\s*")


class EmotionFilter:
    """Pulls [happy]-style mood tags and [shrug]-style gesture tags out of a
    token stream.

    feed() returns the text without tags; `self.current` is the latest mood
    seen and `take_gesture()` returns the latest gesture not yet used. Unknown
    words in brackets are dropped too (never spoken). A "[" at the end of a
    token is held back until we know whether it's a tag.
    """

    MAX_TAG = 22

    def __init__(self, default: str = "neutral"):
        self.current = default
        self.gesture: str | None = None
        self.pending = ""

    def take_gesture(self) -> str | None:
        g, self.gesture = self.gesture, None
        return g

    def feed(self, text: str) -> str:
        self.pending += text
        out = []
        while self.pending:
            m = _TAG.search(self.pending)
            if m:
                out.append(self.pending[:m.start()])
                word = " ".join(m.group(1).lower().replace("-", " ").split())
                gesture = _GESTURE_ALIASES.get(word, word)
                if gesture in GESTURES:
                    self.gesture = gesture
                    if gesture == "laugh":
                        self.current = "happy"
                word = _EMOTION_ALIASES.get(word, word)
                if word in EMOTIONS:
                    self.current = word
                self.pending = self.pending[m.end():]
                continue
            # Hold back a possible unfinished tag at the end.
            i = self.pending.rfind("[")
            if i >= 0 and "]" not in self.pending[i:] and len(self.pending) - i <= self.MAX_TAG:
                out.append(self.pending[:i])
                self.pending = self.pending[i:]
            else:
                out.append(self.pending)
                self.pending = ""
            break
        return "".join(out)

    def flush(self) -> str:
        rest, self.pending = self.pending, ""
        return rest


def clean_for_speech(text: str) -> str:
    """Strip markdown and emoji-ish symbols the TTS would read literally."""
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)
    return re.sub(r"\s+", " ", text).strip()
