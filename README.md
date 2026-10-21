# Talking Portrait Backend

A FastAPI server that lets people talk to painted portraits. Fully open-source models:

| Stage | Model | Runs on |
|---|---|---|
| Speech-to-text | Whisper large-v3-turbo (faster-whisper) | GPU or CPU |
| Brain | Qwen3-8B via Ollama (any OpenAI-compatible server works) | GPU or CPU |
| Voice | Kokoro-82M | GPU or CPU |
| Face | MuseTalk 1.5 (stub to fill in, see `app/lipsync.py`) | GPU |

## How a turn works

```
browser ──text or mic audio──▶ WebSocket /ws/{persona}
                                   │
                    audio? ──▶ Whisper ──▶ transcript
                                   │
                         Qwen3 streams tokens ──▶ sentence splitter
                                   │                 │
                                   │          sentence 1 ──▶ Kokoro ──▶ (MuseTalk) ──▶ browser plays
                                   │          sentence 2 ──▶ Kokoro ──▶ (MuseTalk) ──▶ queued
                                   ▼
                              "done" + full text
```

The LLM and TTS run at the same time: the portrait starts speaking sentence 1
while sentence 2 is still being written. Sending a new message or `stop`
interrupts the current reply (barge-in).

## Setup

```bash
# 1. System packages
sudo apt install espeak-ng ffmpeg          # macOS: brew install espeak-ng ffmpeg

# 2. Python environment (Python 3.10–3.12)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
#   NVIDIA GPU: install the CUDA build of PyTorch first, see pytorch.org

# 3. The LLM
#   install Ollama from https://ollama.com, then:
ollama pull qwen3:8b

# 4. Settings
cp .env.example .env        # set DEVICE=cpu if you have no NVIDIA GPU
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 for the test console: pick a portrait, connect,
type or hold the mic button to talk. The first start downloads the Whisper and
Kokoro weights (~2 GB) and takes a minute.

## Add your own portrait

Add an entry to `personas.json`:

```json
{
  "id": "astronomer",
  "name": "The Astronomer",
  "system_prompt": "You are the painted portrait of ...",
  "voice": "am_michael",
  "portrait": "portraits/astronomer.png",
  "greeting": "Have you seen the comet tonight?",
  "mouth": {"x": 44, "y": 55, "w": 12, "h": 9}
}
```

`mouth` is optional and only used for PNG/JPG portraits: it marks where the
mouth is (in % of the image) so the page can move the jaw while the portrait
speaks. Portraits work best in a 4:5 (tall) shape.

Voices: the first letter is the language (`a` US English, `b` UK English) and
the second is gender (`f`/`m`). Examples: `af_heart`, `af_bella`, `am_michael`,
`bf_emma`, `bm_george`, `bm_fable`. See Kokoro's VOICES list for all of them.

## The avatar (browser side)

The test page shows each character as a framed painting and animates it with
plain HTML/CSS/JS, no GPU needed:

- **Lips** open and close with the loudness of the voice, measured live with the
  Web Audio API. **Eyes** blink every few seconds.
- **Depth:** the frame sways very slowly; inside the painting the background
  and figure move by different amounts (parallax), with a moving varnish sheen.
- Talking makes the frame glow; disconnected portraits look faded.

SVG portraits animate best: put them in `portraits/` and give the moving parts
these classes: `mouth` (opens downward), `lids` (shown briefly to blink),
`backdrop` and `figure` (parallax layers). The three included SVGs are
placeholders to replace with real art. PNG/JPG portraits get the tilt plus a
moving jaw if `mouth` is set in `personas.json`.

**3D characters:** a persona can have a `model` instead of an image. Mr. Hollis
the Butler uses `Business_Male_04` from [Microsoft Rocketbox](https://github.com/microsoft/Microsoft-Rocketbox)
(MIT license, copy in `portraits/models/butler/LICENSE.md`), rendered with
Three.js by `static/avatar3d.js`:

- **Lip-sync:** Kokoro reports every phoneme and how long it lasts;
  `app/visemes.py` turns that into a mouth-shape timeline sent with each
  `speech` message (`"visemes": [[seconds, "aa"], ...]`), and the model follows
  it against the audio's playback time.
- **Body and hands:** motion-captured Rocketbox clips in
  `portraits/models/butler/animations.json`: idle breathing while listening,
  talking gestures while speaking, sometimes a nod afterwards.
- **Face:** natural blinking (sometimes double), small eye movements and
  glances aside, eyebrow lifts and head nods on stressed syllables, and a
  half-asleep look when disconnected. Any other Rocketbox avatar works the same
way: copy its `Export/*_facial.fbx`, convert the head/body textures to JPG/PNG
and add a `model` entry like the butler's.

## Tests

```bash
pytest
```

The tests replace every model with a fake, so they run in a second with no GPU.

## Next steps

1. **Full 3D characters:** replace the paintings with real 3D heads rendered in
   the browser with Three.js and `@pixiv/three-vrm` (VRM models). VRM models
   have mouth shapes (aa, ih, ou, ee, oh) for proper lip shapes, blinking, head
   movement and eyes that follow the visitor; drive the mouth from the same
   audio-level code in `static/index.html`, or send per-sentence phoneme timings
   from Kokoro for exact lip-sync. Runs on a Mac, no server GPU needed. The main
   work is getting three models that fit the castle style (most free VRM
   models are anime-style, so plan to buy or commission them).
2. **Lip-sync video:** implement `MuseTalkLipSync` in `app/lipsync.py`, then set
   `LIPSYNC_ENGINE=musetalk`. Each `speech` message will then include an MP4.
3. **Idle animation:** pre-render blinking/breathing loops per portrait (e.g.
   with LivePortrait) and play them in the frontend between replies.
4. **Faster STT:** swap Whisper for Parakeet TDT 0.6B in `app/stt.py` (English).
5. **Lower latency:** stream raw audio over WebRTC (Pipecat or LiveKit) instead
   of base64 WAV per sentence.

## Files

```
app/main.py        WebSocket server and the streaming pipeline
app/stt.py         Whisper speech-to-text
app/llm.py         streaming chat client (Ollama / vLLM / LM Studio)
app/tts.py         Kokoro text-to-speech
app/lipsync.py     lip-sync interface + MuseTalk stub
app/personas.py    characters and conversation memory
app/text_utils.py  sentence splitting, <think> removal, speech cleanup
app/visemes.py     phonemes -> mouth-shape timeline for 3D lip-sync
static/index.html  browser test console + animated portrait (lips, blink, 3D tilt)
static/avatar3d.js 3D character renderer (Three.js)
portraits/         portrait images (placeholder SVGs) and models/ (3D characters)
personas.json      the characters
```
