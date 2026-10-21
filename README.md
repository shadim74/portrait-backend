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
while sentence 2 is still being written. To start even sooner, the first piece
of a reply is cut at its first natural pause (a comma, dash or short
sentence), and very long sentences are split at commas, so speech typically
begins about 1 second after you finish talking or typing. Sending a new message or `stop`
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

Open http://localhost:8000: you're in the **portrait hall** of Ashford House, with
the three portraits hanging on the wall. Click one (or use ← / →) to walk up
to it; it comes to life and greets you, and the chat opens. Type, hold the
mic button, or turn on hands-free to talk. "Step back" (or Esc) returns to
the whole wall.

Only the portrait you're standing at runs live in 3D; the others show a
pre-rendered still (`portraits/stills/`), so the page stays light. The wall,
panelling and floor use small CC0 textures from [Poly Haven](https://polyhaven.com)
in `static/gallery/`. The other paintings on the wall are public-domain works
(Inness, Bierstadt, Ruysch) via Wikimedia Commons, listed in
`static/gallery/art/CREDITS.md`.

**Hands-free:** press "Hands-free" to talk without holding the button. A
voice detector (Silero VAD, via `@ricky0123/vad-web`) runs in the browser,
sends each thing you say when you pause, and you can interrupt the portrait
just by speaking. With speakers, sound picked up while the portrait talks is
ignored unless it is clearly someone speaking, so it doesn't answer itself;
headphones work best. The first start downloads the Whisper and
Kokoro weights (~2 GB) and takes a minute.

## The characters

Three people around **Ashford House**, an old family house that's now a small
museum. Each one believes they're a real person talking face to face with the
visitor (never a portrait, character or AI), talks in casual modern English,
and knows the other two:

- **Hollis, the House Manager** (default): has run the house for 25
  years, practical, dry humour, runs on coffee, fights the heating.
- **Arthur, the Old Headmaster**: retired village headteacher who volunteers
  as a guide, warm and jokey, loves puzzles, biscuits and football.
- **Clara, the Tour Guide**: 24, great-granddaughter of the family that
  built the house, gives tours (sometimes dressed as a 1912 lady's maid), a
  cheerful history nerd obsessed with the house's Downton-era past, sitting on
  a secret from her great-grandmother Eleanor's diaries.

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

## Chat history

Conversations are saved in `data/history.db` (SQLite; change with
`HISTORY_DB`, or set it empty to save nothing). Each browser keeps a random
visitor id, so when you come back the portrait shows your earlier
conversation, greets you with its `welcome_back` line and remembers what was
said (the LLM sees the last `MAX_HISTORY` messages). "Clear history" on the
test page forgets the conversation with that portrait.

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

**3D characters:** a persona can have a `model` instead of an image. All three
characters are 3D models from [Microsoft Rocketbox](https://github.com/microsoft/Microsoft-Rocketbox)
(MIT license, copy in each `portraits/models/<name>/LICENSE.md`), rendered with
Three.js by `static/avatar3d.js`:

| Character | Rocketbox model | Folder |
|---|---|---|
| Hollis | `Business_Male_04` | `portraits/models/butler/` |
| Arthur | `Male_Adult_03` | `portraits/models/arthur/` |
| Clara | `Female_Adult_07` | `portraits/models/clara/` |

They share one skeleton and the same face controls, so every feature below
works for all of them:


- **Lip-sync:** Kokoro reports every phoneme and how long it lasts;
  `app/visemes.py` turns that into a mouth-shape timeline sent with each
  `speech` message (`"visemes": [[seconds, "aa"], ...]`), and the model follows
  it against the audio's playback time.
- **Body and hands:** motion-captured Rocketbox clips in
  `portraits/models/animations/` (`male.json`, `female.json`): calm breathing
  while listening, and talking gestures in the style of the current mood
  (relaxed, excited, sad or angry).
- **Gestures:** the LLM can add tags like `[wave]`, `[shrug]`, `[laugh]`,
  `[nod]`, `[shake head]` or `[think]`; the server strips them and sends the
  gesture with that speech piece, and the character plays it. Greetings come
  with a wave.
- **Listening:** small nods while you talk (hands-free or hold-to-talk) and a
  nod when you finish.
- **Body language:** now and then while listening, a glance to one side, a
  look over the shoulder, or a slight turn and weight shift.
- **Out of the frame:** the 3D canvas is larger than the picture and sits over
  the gold frame, so gesturing arms and hands reaching toward you come out of
  it, while the lower body stays hidden behind the bottom edge.
- **Expressions:** the LLM starts each reply (and any change of feeling)
  with a tag like `[happy]`, `[sad]`, `[surprised]`, `[angry]` or
  `[thinking]`. The server strips the tags (they're never shown or spoken)
  and sends the mood with each speech piece; the face blends into it as that
  piece plays, then relaxes to the character's resting `mood` from
  `personas.json` a few seconds after he stops talking.
- **Face:** natural blinking (sometimes double), small eye movements and
  glances aside, eyebrow lifts and head nods on stressed syllables, and a
  half-asleep look when disconnected.

Any other Rocketbox avatar works the same way: copy its `Export/*_facial.fbx`,
convert the head/body (and, for hair, opacity) textures to JPG/PNG and add a
`model` entry like the others in `personas.json`.

## Tests

```bash
pytest
```

The tests replace every model with a fake, so they run in a second with no GPU.

## Next steps

1. **Custom 3D characters:** the Rocketbox models are modern everyday people.
   For bespoke looks, commission or buy models with ARKit blend shapes and a
   compatible skeleton, or use VRM models via `@pixiv/three-vrm`.
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
app/history.py     saved chat history (SQLite)
app/text_utils.py  sentence splitting, <think> removal, speech cleanup
app/visemes.py     phonemes -> mouth-shape timeline for 3D lip-sync
static/index.html  the portrait hall: wall, walking between portraits, chat
static/avatar3d.js 3D character renderer (Three.js)
portraits/         portrait images (placeholder SVGs) and models/ (3D characters)
personas.json      the characters
```
