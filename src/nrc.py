"""NRC Emotion Lexicon word-list scoring (Step 5, take #2).

Loads the NRC Word-Emotion Association Lexicon (EmoLex) v0.92 and derives a
review's primary emotion by counting how many of its words are associated with
each of the eight discrete emotions, then taking the highest-scoring emotion.
No model calls are required.
"""
import re
from collections import defaultdict

from . import config

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness",
            "surprise", "trust"]

_token_re = re.compile(r"[a-z0-9']+")

#: word -> set(emotions it is associated with)
_lexicon = None


def load_lexicon():
    """Lazily parse the NRC file into {word: set(emotions)}. Returns the cache."""
    global _lexicon
    if _lexicon is not None:
        return _lexicon
    lex = defaultdict(set)
    with open(config.NRC_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if len(parts) != 3:
                continue
            word, emotion, flag = parts
            if emotion not in EMOTIONS:
                continue  # ignore positive/negative polarity rows here
            if flag == "1":
                lex[word.strip().lower()].add(emotion)
    _lexicon = dict(lex)
    return _lexicon


def score_text(text: str) -> dict:
    """Return {emotion: count} for a piece of text against the lexicon."""
    lex = load_lexicon()
    counts = {e: 0 for e in EMOTIONS}
    for tok in _token_re.findall((text or "").lower()):
        for e in lex.get(tok, ()):
            counts[e] += 1
    return counts


def primary_emotion(text: str):
    """Return the highest-scoring emotion ('joy', 'anger', ...), tie-broken by
    the NRC emotion order. Returns '' if no word matched any emotion."""
    counts = score_text(text)
    top = max(counts, key=lambda e: (counts[e], -EMOTIONS.index(e)))
    return top if counts[top] > 0 else ""
