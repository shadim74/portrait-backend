"""Tests that run without any AI models: fake engines stand in for them."""
import asyncio
import base64

from fastapi.testclient import TestClient

from app.main import Engines, app
from app.personas import Persona
from app.text_utils import SentenceSplitter, ThinkFilter, clean_for_speech


# ---------- pure helpers ----------

def test_splitter_streams_sentences():
    s = SentenceSplitter()
    out = []
    for tok in ["Hello there, trav", "eller. Dr. Smith was ", "here at 3.5 o'clock! And", " then?"]:
        out += s.feed(tok)
    out += s.flush()
    # the first chunk ends at the first pause so speech can start early
    assert out == ["Hello there,", "traveller.", "Dr. Smith was here at 3.5 o'clock!", "And then?"]


def test_think_filter_handles_split_tags():
    f = ThinkFilter()
    text = "".join(f.feed(t) for t in ["<thi", "nk>secret plan</th", "ink>Hello", " world"]) + f.flush()
    assert text == "Hello world"


def test_clean_for_speech():
    assert clean_for_speech("**Ah** _yes_ 🙂 see [this](http://x)") == "Ah yes see this"


# ---------- end-to-end over the WebSocket, with fake models ----------

class FakeSTT:
    def transcribe(self, audio, language=None):
        return "who are you"


class FakeLLM:
    def __init__(self, delay=0.0):
        self.delay = delay

    async def stream(self, messages):
        for tok in ["<think>hmm</think>", "I am the old ", "headmaster. ", "I like ", "sweets very much."]:
            await asyncio.sleep(self.delay)
            yield tok


class FakeTTS:
    def synthesize(self, text, voice="af_heart"):
        return b"RIFF" + text.encode()


class FakeLipSync:
    def render(self, wav, portrait):
        return None


def make_client(llm_delay=0.0):
    app.state.engines = Engines(stt=FakeSTT(), llm=FakeLLM(llm_delay), tts=FakeTTS(), lipsync=FakeLipSync())
    app.state.personas = {"headmaster": Persona(id="headmaster", name="Headmaster",
                                                system_prompt="You are a headmaster.",
                                                greeting="Welcome, visitor!")}
    return TestClient(app)


def receive_until_done(ws):
    msgs = []
    while True:
        m = ws.receive_json()
        msgs.append(m)
        if m["type"] in ("done", "error"):
            return msgs


def test_splitter_starts_early_on_first_clause():
    s = SentenceSplitter()
    out = []
    for tok in ["Ah, the library, my dear visitor, ", "is where the old books whisper. ", "Come", " and see."]:
        out += s.feed(tok)
    out += s.flush()
    # The first two chunks are cut at the first pause so speech can start right
    # away; a very short first clause ("Ah,") stays attached.
    assert out == ["Ah, the library,", "my dear visitor,", "is where the old books whisper.", "Come and see."]


def test_splitter_early_chunks_without_pause_cut_after_words():
    s = SentenceSplitter()
    words = "Well then I suppose the oldest thing in this house would have to be the clock in the hall".split()
    out = []
    for w in words:
        out += s.feed(w + " ")
    assert out[:2] == ["Well then I suppose", "the oldest thing in this house would have"]
    assert " ".join(out + s.flush()) == " ".join(words)


def test_splitter_cuts_long_sentences_at_clauses():
    s = SentenceSplitter()
    s.emitted = 2  # past the early chunks
    long = ("The portrait of my late wife hung in the east wing for many years, "
            "until the great storm of the winter tore the roof clean away.")
    out = []
    for word in long.split(" "):       # tokens arrive a word at a time, like the LLM stream
        out += s.feed(word + " ")
    out += s.flush()
    assert out == ["The portrait of my late wife hung in the east wing for many years,",
                   "until the great storm of the winter tore the roof clean away."]


def test_text_turn_streams_sentences_and_audio():
    with make_client() as client, client.websocket_connect("/ws/headmaster") as ws:
        assert ws.receive_json()["type"] == "ready"
        greeting = receive_until_done(ws)
        assert greeting[0]["type"] == "speech" and greeting[0]["text"] == "Welcome, visitor!"

        ws.send_json({"type": "text", "text": "Who are you?"})
        msgs = receive_until_done(ws)
        speech = [m for m in msgs if m["type"] == "speech"]
        # speech starts after the first few words, then follows the sentences
        assert [m["text"] for m in speech] == ["I am the old", "headmaster.", "I like sweets very much."]
        assert base64.b64decode(speech[0]["audio"]).startswith(b"RIFF")
        assert "hmm" not in msgs[-1]["text"]          # hidden reasoning never leaks
        assert msgs[-1]["text"] == "I am the old headmaster. I like sweets very much."


def test_audio_turn_transcribes_first():
    with make_client() as client, client.websocket_connect("/ws/headmaster") as ws:
        ws.receive_json(); receive_until_done(ws)   # ready + greeting
        ws.send_json({"type": "audio", "audio": base64.b64encode(b"fake-webm").decode()})
        msgs = receive_until_done(ws)
        assert msgs[0] == {"type": "transcription", "text": "who are you"}
        assert any(m["type"] == "speech" for m in msgs)


def test_barge_in_interrupts_reply():
    with make_client(llm_delay=0.2) as client, client.websocket_connect("/ws/headmaster") as ws:
        ws.receive_json(); receive_until_done(ws)
        ws.send_json({"type": "text", "text": "Tell me a long story"})
        ws.receive_json()                             # first token arrives
        ws.send_json({"type": "stop"})
        seen = set()
        while "interrupted" not in seen:
            seen.add(ws.receive_json()["type"])
        assert "done" not in seen


def test_unknown_persona():
    with make_client() as client, client.websocket_connect("/ws/nobody") as ws:
        assert ws.receive_json()["type"] == "error"
