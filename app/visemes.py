"""Turn Kokoro's phonemes + durations into a mouth-shape timeline for lip-sync.

Kokoro (misaki) writes one character per phoneme and predicts how many frames
each lasts (1 frame = 600 samples at 24 kHz = 25 ms). We map every phoneme to
one of the 15 Oculus visemes, which most 3D characters (Rocketbox, ARKit/
Ready Player Me style rigs) have as blend shapes.
"""

FRAME_SECONDS = 600 / 24_000

# Oculus viseme names: sil PP FF TH DD kk CH SS nn RR aa E I O U
_GROUPS = {
    "PP": "pbm",
    "FF": "fv",
    "TH": "θð",
    "DD": "tdɾT",
    "kk": "kgŋɡx",
    "CH": "ʧʤʃʒ",
    "SS": "sz",
    "nn": "nlɫ",
    "RR": "ɹrɻ",
    "aa": "ɑaæʌɐIW",       # I = "eye" (aɪ), W = "ow" (aʊ) start open
    "E": "ɛeAəɚɜᵊh",       # A = "ay" (eɪ)
    "I": "iɪj",
    "O": "ɔoOQɒY",         # O/Q = "oh" (oʊ / əʊ), Y = "oy" (ɔɪ)
    "U": "uʊw",
}
PHONEME_TO_VISEME = {ch: vis for vis, chars in _GROUPS.items() for ch in chars}
# Stress and length marks don't change the mouth: they extend the previous sound.
_CONTINUE = set("ˈˌː")


def viseme_for(phoneme: str) -> str | None:
    """Viseme for one phoneme character; None = keep the previous one."""
    if phoneme in _CONTINUE:
        return None
    return PHONEME_TO_VISEME.get(phoneme, "sil")  # spaces, punctuation, unknown


def timeline(phonemes: str, durations: list[int], start: float = 0.0) -> list[list]:
    """[[seconds, viseme], ...] for one Kokoro chunk.

    durations is Kokoro's pred_dur: one entry per phoneme plus a leading
    <bos> and trailing <eos> frame count.
    """
    if not phonemes or len(durations) != len(phonemes) + 2:
        return []
    out: list[list] = []
    t = start + durations[0] * FRAME_SECONDS
    for ph, frames in zip(phonemes, durations[1:-1]):
        vis = viseme_for(ph)
        if vis is not None and (not out or out[-1][1] != vis):
            out.append([round(t, 3), vis])
        t += frames * FRAME_SECONDS
    if out and out[-1][1] != "sil":
        out.append([round(t, 3), "sil"])
    return out
