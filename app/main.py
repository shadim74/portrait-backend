"""Talking-portrait backend.

One WebSocket per visitor:  ws://host:8000/ws/{persona_id}

Client -> server (JSON):
  {"type": "text",  "text": "Hello!"}
  {"type": "audio", "audio": "<base64 webm/wav>", "language": "en"}   # optional language
  {"type": "stop"}                                                    # interrupt reply

Server -> client (JSON):
  {"type": "ready",         "persona": {...}}
  {"type": "transcription", "text": "..."}            # what we heard
  {"type": "token",         "text": "..."}            # live reply text
  {"type": "speech", "index": 0, "text": "...", "audio": "<base64 wav>", "video": "<base64 mp4>|null"}
  {"type": "done",          "text": "<full reply>"}
  {"type": "interrupted"}
  {"type": "error",         "message": "..."}

The reply is streamed sentence by sentence: while the LLM writes sentence 2,
sentence 1 is already being turned into speech (and video), so the portrait
starts talking long before the full answer exists.
"""
import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .personas import Conversation, Persona, load_personas
from .text_utils import SentenceSplitter, ThinkFilter, clean_for_speech

log = logging.getLogger("portrait")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).parent.parent
STATIC = ROOT / "static"
PORTRAITS = ROOT / "portraits"


def asset_url(rel_path: str) -> str | None:
    """Browser URL for a file under portraits/, or None if it isn't there."""
    if not rel_path:
        return None
    path = (ROOT / rel_path).resolve()
    if path.is_file() and path.is_relative_to(PORTRAITS.resolve()):
        return "/portraits/" + path.relative_to(PORTRAITS.resolve()).as_posix()
    return None


def persona_info(p: Persona) -> dict:
    model = None
    if p.model and (url := asset_url(p.model.get("file", ""))):
        model = {**p.model, "url": url}
    return {"id": p.id, "name": p.name, "greeting": p.greeting,
            "portrait": asset_url(p.portrait), "mouth": p.mouth, "model": model}


@dataclass
class Engines:
    stt: object      # .transcribe(bytes, language) -> str
    llm: object      # .stream(messages) -> async iterator of str
    tts: object      # .synthesize(text, voice) -> wav bytes
    lipsync: object  # .render(wav bytes, portrait path) -> mp4 bytes | None


def load_engines() -> Engines:
    from .lipsync import make_lipsync
    from .llm import ChatModel
    from .stt import SpeechToText
    from .tts import TextToSpeech

    t = time.perf_counter()
    engines = Engines(stt=SpeechToText(), llm=ChatModel(), tts=TextToSpeech(), lipsync=make_lipsync())
    log.info("models loaded in %.1fs", time.perf_counter() - t)
    return engines


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "engines"):      # tests inject fake engines
        app.state.engines = await asyncio.to_thread(load_engines)
    if not hasattr(app.state, "personas"):
        app.state.personas = load_personas()
    # GPU models aren't safe to call from several threads at once; one lock each.
    app.state.gpu_locks = {name: asyncio.Lock() for name in ("stt", "tts", "lipsync")}
    yield


app = FastAPI(title="Talking Portrait Backend", lifespan=lifespan)
app.mount("/portraits", StaticFiles(directory=PORTRAITS, check_dir=False), name="portraits")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def test_page():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/personas")
async def personas():
    return [persona_info(p) for p in app.state.personas.values()]


class Session:
    """One visitor talking to one portrait."""

    def __init__(self, ws: WebSocket, persona: Persona):
        self.ws = ws
        self.conv = Conversation(persona)
        self.engines: Engines = ws.app.state.engines
        self.locks = ws.app.state.gpu_locks
        self.send_lock = asyncio.Lock()
        self.turn: asyncio.Task | None = None

    async def send(self, **msg):
        async with self.send_lock:
            await self.ws.send_json(msg)

    async def run_blocking(self, lock_name, fn, *args):
        async with self.locks[lock_name]:
            return await asyncio.to_thread(fn, *args)

    async def speak(self, index: int, text: str):
        persona = self.conv.persona
        tts = self.engines.tts
        if hasattr(tts, "synthesize_timed"):   # also returns mouth shapes for 3D lip-sync
            wav, visemes = await self.run_blocking("tts", tts.synthesize_timed, text, persona.voice)
        else:
            wav, visemes = await self.run_blocking("tts", tts.synthesize, text, persona.voice), None
        video = await self.run_blocking("lipsync", self.engines.lipsync.render, wav, persona.portrait)
        await self.send(
            type="speech", index=index, text=text,
            audio=base64.b64encode(wav).decode(),
            video=base64.b64encode(video).decode() if video else None,
            visemes=visemes,
        )

    async def reply(self, user_text: str):
        """LLM -> sentences -> TTS (+ lip-sync), running concurrently."""
        started = time.perf_counter()
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        full_reply: list[str] = []

        async def produce():
            splitter = SentenceSplitter()
            think = ThinkFilter()  # hidden reasoning must never be spoken

            async def emit(text: str):
                if not text:
                    return
                full_reply.append(text)
                await self.send(type="token", text=text)
                for s in splitter.feed(text):
                    await sentences.put(s)

            try:
                async for token in self.engines.llm.stream(self.conv.messages_for(user_text)):
                    await emit(think.feed(token))
                await emit(think.flush())
                for s in splitter.flush():
                    await sentences.put(s)
            finally:
                await sentences.put(None)

        async def consume():
            index = 0
            while (sentence := await sentences.get()) is not None:
                spoken = clean_for_speech(sentence)
                if not spoken:
                    continue
                await self.speak(index, spoken)
                if index == 0:
                    log.info("first audio after %.0f ms", (time.perf_counter() - started) * 1000)
                index += 1

        try:
            await asyncio.gather(produce(), consume())
        finally:
            # Remember what was said, even if the visitor interrupted midway.
            self.conv.add_reply("".join(full_reply).strip())
        await self.send(type="done", text="".join(full_reply).strip())

    async def handle_turn(self, msg: dict):
        try:
            if msg["type"] == "audio":
                audio = base64.b64decode(msg["audio"])
                text = await self.run_blocking("stt", self.engines.stt.transcribe, audio, msg.get("language"))
                await self.send(type="transcription", text=text)
                if not text:
                    return
            else:
                text = (msg.get("text") or "").strip()
                if not text:
                    return
            await self.reply(text)
        except asyncio.CancelledError:
            await self.send(type="interrupted")
            raise
        except Exception as e:  # keep the connection alive on model errors
            log.exception("turn failed")
            await self.send(type="error", message=str(e))

    async def cancel_turn(self):
        if self.turn and not self.turn.done():
            self.turn.cancel()
            try:
                await self.turn
            except (asyncio.CancelledError, Exception):
                pass

    async def greet(self):
        if self.conv.persona.greeting:
            self.conv.add_reply(self.conv.persona.greeting)
            await self.speak(0, self.conv.persona.greeting)
            await self.send(type="done", text=self.conv.persona.greeting)


@app.websocket("/ws/{persona_id}")
async def portrait_socket(ws: WebSocket, persona_id: str):
    await ws.accept()
    persona = ws.app.state.personas.get(persona_id)
    if not persona:
        await ws.send_json({"type": "error", "message": f"unknown persona '{persona_id}'"})
        await ws.close()
        return

    session = Session(ws, persona)
    await session.send(type="ready", persona=persona_info(persona))
    session.turn = asyncio.create_task(session.greet())
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") in ("text", "audio", "stop"):
                await session.cancel_turn()          # new input = barge-in
            if msg.get("type") in ("text", "audio"):
                session.turn = asyncio.create_task(session.handle_turn(msg))
    except WebSocketDisconnect:
        pass
    finally:
        await session.cancel_turn()
