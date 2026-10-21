from app.visemes import FRAME_SECONDS, timeline, viseme_for


def test_phoneme_mapping():
    assert viseme_for("m") == "PP"
    assert viseme_for("f") == "FF"
    assert viseme_for("ð") == "TH"
    assert viseme_for("ʊ") == "U"
    assert viseme_for("O") == "O"       # misaki "oh"
    assert viseme_for(" ") == "sil"
    assert viseme_for(",") == "sil"
    assert viseme_for("ˈ") is None      # stress mark: keep previous shape


def test_timeline_timing_and_merging():
    # "mˈɑ" + space: <bos>=2 frames, m=2, ˈ=1, ɑ=4, space=2, <eos>=1
    tl = timeline("mˈɑ ", [2, 2, 1, 4, 2, 1], start=1.0)
    f = FRAME_SECONDS
    assert tl == [
        [round(1.0 + 2 * f, 3), "PP"],
        [round(1.0 + 5 * f, 3), "aa"],      # stress mark extended the "m"
        [round(1.0 + 9 * f, 3), "sil"],
    ]


def test_timeline_rejects_mismatched_lengths():
    assert timeline("abc", [1, 2]) == []
