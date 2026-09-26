#!/usr/bin/env python
"""Load Amazon 2023 Gift_Cards reviews, build a clean dataframe, run EDA."""
import json
import pandas as pd
import numpy as np

RAW = "Gift_Cards.jsonl"
OUT_CLEAN = "gift_cards_reviews.parquet"
OUT_CSV = "gift_cards_reviews.csv"

# 1) Load JSONL into a dataframe
records = []
with open(RAW, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

df = pd.DataFrame(records)
print(f"Loaded {len(df):,} reviews")

# 2) Clean / type-cast
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")   # ms -> datetime
df["rating"] = df["rating"].astype(float).round(0).astype(int)  # ratings are integral
df["helpful_vote"] = df["helpful_vote"].fillna(0).astype(int)
df["verified_purchase"] = df["verified_purchase"].astype(bool)
df["n_images"] = df["images"].apply(len)
# text length (chars) and word count as useful derived features
df["text_len"] = df["text"].astype(str).str.len()
df["word_count"] = df["text"].astype(str).str.split().str.len()
df["year"] = df["timestamp"].dt.year

# Save reusable versions (date as ISO string in CSV for portability)
csv_df = df.copy()
csv_df["timestamp"] = csv_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
df.to_parquet(OUT_CLEAN, index=False)
csv_df.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CLEAN} ({df.shape[0]:,} x {df.shape[1]})")
print(f"Saved -> {OUT_CSV}\n")

# 3) --- EDA summary ---
print("=" * 60)
print("BASIC SHAPE & TYPES")
print("=" * 60)
print(df.dtypes.to_string())
print(f"\nNull values per column:\n{df.isnull().sum().to_string()}")

print("\n" + "=" * 60)
print("RATINGS DISTRIBUTION")
print("=" * 60)
print(df["rating"].value_counts().sort_index().to_string())
print(f"\nMean rating: {df['rating'].mean():.3f} | Median: {df['rating'].median():.1f}")

print("\n" + "=" * 60)
print("PRODUCT / REVIEWER COVERAGE")
print("=" * 60)
print(f"Unique ASINs:     {df['asin'].nunique():,}")
print(f"Unique users:     {df['user_id'].nunique():,}")
print(f"Avg reviews/item: {len(df)/df['asin'].nunique():.1f}")
reviews_per_user = df["user_id"].value_counts()
print(f"Reviews/user:     mean={reviews_per_user.mean():.1f} median={reviews_per_user.median():.0f} max={reviews_per_user.max()}")

print("\n" + "=" * 60)
print("VERIFIED PURCHASE")
print("=" * 60)
print(df["verified_purchase"].value_counts().to_string())
print(f"Verified rate: {df['verified_purchase'].mean()*100:.1f}%")

print("\n" + "=" * 60)
print("HELPFUL VOTES")
print("=" * 60)
print(f"Votes>0: {(df['helpful_vote']>0).sum():,}  ({(df['helpful_vote']>0).mean()*100:.2f}%)")
print(f"Votes>0 max: {df['helpful_vote'].max()}")
print(df["helpful_vote"].describe().to_string())

print("\n" + "=" * 60)
print("TIME RANGE")
print("=" * 60)
print(f"Earliest: {df['timestamp'].min()}")
print(f"Latest:   {df['timestamp'].max()}")
print(df["year"].value_counts().sort_index().to_string())

print("\n" + "=" * 60)
print("REVIEW LENGTH (chars)")
print("=" * 60)
print(df["text_len"].describe().to_string())

print("\n" + "=" * 60)
print("IMAGES")
print("=" * 60)
print(f"Reviews with >=1 image: {(df['n_images']>0).sum():,}")

print("\n" + "=" * 60)
print("TOP 10 MOST-REVIEWED PRODUCTS")
print("=" * 60)
print(df["asin"].value_counts().head(10).to_string())

print("\n" + "=" * 60)
print("TOP REVIEW TEXT SNIPPETS (most helpful)")
print("=" * 60)
top = df.nlargest(5, "helpful_vote")
for _, r in top.iterrows():
    print(f"[{int(r['rating'])}★ | {r['helpful_vote']} helpful | {r['verified_purchase']}] {r['title']}")
    print(f"   {(str(r['text'])[:220])}")
