"""The reusable review-classification prompt and its response parser.

A review is judged ONLY on its own title + body text. The star rating is never
shown to the model; it is used only afterwards as the ground-truth label.
"""

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness",
            "surprise", "trust"]

_SENTIMENT_BINARY = ("POSITIVE (enthusiastic, satisfied, recommends)\n"
                     "- NEGATIVE (unhappy, disappointed, complains, warns others)")
_SENTIMENT_THREE = ("POSITIVE (enthusiastic, satisfied, recommends)\n"
                    "- NEUTRAL (matter-of-fact, mixed, ambivalent, no strong feeling either way)\n"
                    "- NEGATIVE (unhappy, disappointed, complains, warns others)")


def build_prompt(title: str, body: str, *, three_class: bool = False,
                 include_emotion: bool = False) -> str:
    """Return the instruction prompt for one review.

    three_class=True  -> {POSITIVE, NEUTRAL, NEGATIVE}
    include_emotion=True -> also asks the model to output a primary emotion,
                            then compares against the NRC word-list derivation.
    """
    sentiment_opts = _SENTIMENT_THREE if three_class else _SENTIMENT_BINARY

    emotion_instruction = (
        "\n\nFinally, decide the review's PRIMARY EMOTION from exactly one of: "
        + ", ".join(f"'{e}'" for e in EMOTIONS) + "."
        if include_emotion else ""
    )
    output_keys = '"sentiment": "SENTIMENT"' + (', "emotion": "emotion"' if include_emotion else "")
    output_example = '{"sentiment": "POSITIVE"' + (', "emotion": "joy"' if include_emotion else "") + "}"

    prompt = f"""You are a product-review sentiment analyst. Read the Amazon review below and
classify its SENTIMENT based ONLY on the review's own title and body text.
You do not and must not consider any star rating.

Review TITLE: {title}
Review BODY: {body}

Choose the SENTIMENT as exactly one of:
- {sentiment_opts}

Handle edge cases yourself: if the title and body conflict, weight the body's
explicit opinions more heavily than the title's headline; terse or angry
short reviews should still be classified by their tone.{emotion_instruction}

Respond with a single JSON object on one line, and NOTHING else. Format:
{output_example}
Use uppercase SENTIMENT and lowercase emotion."""
    return prompt


def parse_response(content: str, *, include_emotion: bool = False):
    """Extract sentiment (and optionally emotion) from a model reply, tolerant
    of the model being slightly sloppy (no strict JSON required)."""
    import json as _json
    sentiment = None
    emotion = None

    text = (content or "").strip()
    # Try strict JSON first
    try:
        obj = _json.loads(text)
        if isinstance(obj, dict):
            sentiment = obj.get("sentiment")
            emotion = obj.get("emotion")
    except Exception:
        pass

    # Fallback: scan the raw text for the known labels / tokens
    if sentiment not in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
        for tok in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
            if tok in text.upper():
                sentiment = tok
                break
    if include_emotion:
        if emotion not in EMOTIONS:
            low = text.lower()
            for e in EMOTIONS:
                if e in low:
                    emotion = e
                    break
    return sentiment, emotion
