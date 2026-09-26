#!/usr/bin/env python
"""Run the LLM classification over a (sub)set of reviews and save results.

Modes:
  spotcheck   -- a handful of obviously positive/negative reviews (Step 1)
  batch1      -- first 100 rows in order, BINARY (Step 2)
  balanced    -- ~50/class balanced sample, THREE-CLASS + emotions (Step 5/6)

Results are cached on disk so an interrupted run can be resumed without
re-calling the model for already-scored reviews.
"""
import argparse
import json
import os
import sys

import pandas as pd

from . import config
from . import data as datamod
from . import llm
from . import nrc
from .prompt import EMOTIONS

CACHE_DIR = os.path.join(config.OUT_DIR, "_cache")


def _cache_path(mode: str) -> str:
    return os.path.join(CACHE_DIR, f"cache_{mode}.json")


def _load_cache(mode: str) -> dict:
    p = _cache_path(mode)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def _save_cache(mode: str, cache: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = _cache_path(mode) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, _cache_path(mode))


def classify_df(df, *, three_class: bool, include_emotion: bool,
                mode: str):
    """Add model_sentiment / llm_emotion / nrc_emotion / nrc_scores columns."""
    cache = _load_cache(mode)
    dump_every = 10
    rows = []
    for idx, r in df.iterrows():
        key = f"{r['title']}\x00{r['text']}"
        rec = cache.get(key)
        if rec is None:
            sent, emo = llm.classify(r["title"], r["text"],
                                     three_class=three_class,
                                     include_emotion=include_emotion)
            rec = {"sentiment": sent, "emotion": emo}
            cache[key] = rec
            if len(cache) % dump_every == 0:
                _save_cache(mode, cache)
        nrc_scores = nrc.score_text(r["text"]) if include_emotion else {}
        nrc_emo = nrc.primary_emotion(r["text"]) if include_emotion else ""
        rows.append(dict(r) | {
            "model_sentiment": rec["sentiment"],
            "llm_emotion": rec.get("emotion", ""),
            "nrc_emotion": nrc_emo,
            **{f"nrc_{k}": v for k, v in nrc_scores.items()},
        })
        sys.stdout.write(f"\rscored {len(rows)}/{len(df)}")
        sys.stdout.flush()
    _save_cache(mode, cache)
    print()
    return pd.DataFrame(rows)


def metrics(df, class_names, label_col="_correct"):
    """Overall + per-class accuracy and a confusion matrix."""
    total = len(df)
    correct = (df["model_sentiment"] == df[label_col]).sum()
    overall = correct / total if total else 0
    per_class = {}
    matrix = {gt: {pr: 0 for pr in class_names} for gt in class_names}
    for gt, pr in zip(df[label_col], df["model_sentiment"]):
        matrix[gt][pr] += 1
    for cls in class_names:
        sub = df[df[label_col] == cls]
        ncls = len(sub)
        ccls = int((sub["model_sentiment"] == cls).sum())
        per_class[cls] = {
            "n": ncls,
            "correct": ccls,
            "accuracy": (ccls / ncls) if ncls else 0,
        }
    return {"n": total, "correct": int(correct), "overall_accuracy": float(overall),
            "per_class": per_class, "confusion": matrix}


def run(mode: str):
    df = datamod.load_reviews()
    if mode == "spotcheck":
        picks = pd.concat([
            df[df["rating"] >= 4].head(3),
            df[df["rating"] <= 2].head(3),
        ])
        out = classify_df(picks, three_class=False, include_emotion=False, mode=mode)
        for _, r in out.iterrows():
            print(f"[{r['rating']}*] {r['title']!r} -> {r['model_sentiment']}")
        return

    if mode == "batch1":
        sub = df.head(config.BATCH1_N).copy()
        sub["_correct"] = sub["rating"].apply(config.rating_to_binary)
        addn = sub.drop(columns=["images"])
        res = classify_df(addn, three_class=False, include_emotion=False, mode=mode)
        out_path = os.path.join(config.OUT_DIR, "batch1_100.csv")
        res.insert(0, "ground_truth", res.pop("_correct"))
        cols = ["asin", "rating", "year", "verified_purchase", "helpful_vote",
                "ground_truth", "model_sentiment", "title", "text"]
        res = res[[c for c in cols if c in res.columns]]
        res.insert(6, "correct", res["ground_truth"] == res["model_sentiment"])
        res.to_csv(out_path, index=False)
        m = metrics(res, ["POSITIVE", "NEGATIVE"], label_col="ground_truth")
        metadata = {"mode": mode, "n": m["n"], "overall_accuracy": m["overall_accuracy"],
                    "per_class": m["per_class"], "confusion": m["confusion"]}
        with open(os.path.join(config.OUT_DIR, "batch1_100_metrics.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        return res, m, out_path

    if mode == "balanced":
        sub = datamod.balanced_sample(df, config.BALANCED_PER_CLASS, config.SEED).copy()
        sub["_correct"] = sub["rating"].apply(config.rating_to_three)
        addn = sub.drop(columns=["images", "_class"])
        res = classify_df(addn, three_class=True, include_emotion=True, mode=mode)
        res.insert(0, "ground_truth", res.pop("_correct"))
        cols = ["asin", "rating", "year", "verified_purchase", "helpful_vote",
                "ground_truth", "model_sentiment", "llm_emotion", "nrc_emotion",
                "title", "text"]
        res = res[[c for c in cols if c in res.columns]]
        res.insert(7, "correct", res["ground_truth"] == res["model_sentiment"])
        out_path = os.path.join(config.OUT_DIR, "balanced_150.csv")
        res.to_csv(out_path, index=False)
        m = metrics(res, ["POSITIVE", "NEUTRAL", "NEGATIVE"], label_col="ground_truth")
        metadata = {"mode": mode, "n": m["n"], "overall_accuracy": m["overall_accuracy"],
                    "per_class": m["per_class"], "confusion": m["confusion"]}
        with open(os.path.join(config.OUT_DIR, "balanced_150_metrics.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        return res, m, out_path

    raise ValueError(f"unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["spotcheck", "batch1", "balanced"])
    args = ap.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
