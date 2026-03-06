import pandas as pd
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("data/myntra_products_catalog.csv")

# Combine relevant text fields
df["text"] = (
    df["ProductName"].fillna("") + " " +
    df["Description"].fillna("")
).str.lower()

# ----------------------------
# CATEGORY DETECTION
# ----------------------------

def detect_category(text):
    if re.search(r"\b(shirt|top|kurta|tshirt|blouse|jacket)\b", text):
        return "top"
    elif re.search(r"\b(jeans|trouser|pants|shorts|skirt)\b", text):
        return "bottom"
    elif re.search(r"\b(shoes|heels|sneakers|sandals|boots)\b", text):
        return "footwear"
    else:
        return "other"

df["category"] = df["text"].apply(detect_category)

# ----------------------------
# OCCASION DETECTION
# ----------------------------

def detect_occasion(text):
    if re.search(r"\b(formal|office|business|suit)\b", text):
        return "formal"
    elif re.search(r"\b(party|evening|festive|wedding)\b", text):
        return "party"
    else:
        return "casual"

df["occasion"] = df["text"].apply(detect_occasion)

# ----------------------------
# BODY TYPE FIT (Smart Rule)
# ----------------------------

def detect_body_fit(text):
    if "high waist" in text:
        return "pear"
    elif "slim fit" in text:
        return "rectangle"
    elif "regular fit" in text:
        return "all"
    else:
        return random.choice(["pear", "rectangle", "apple", "all"])

df["body_type_fit"] = df["text"].apply(detect_body_fit)

# ----------------------------
# TREND SCORE (Keyword weighted)
# ----------------------------

trend_keywords = ["oversized", "cropped", "slim fit", "street", "trendy"]

def calculate_trend_score(text):
    score = 5
    for word in trend_keywords:
        if word in text:
            score += 1
    return min(score, 10)

df["trend_score"] = df["text"].apply(calculate_trend_score)

# ----------------------------
# SUSTAINABILITY SCORE
# ----------------------------

sustainable_keywords = ["organic", "linen", "cotton", "eco", "recycled"]

def calculate_sustainability(text):
    score = 5
    for word in sustainable_keywords:
        if word in text:
            score += 1
    return min(score, 10)

df["sustainability_score"] = df["text"].apply(calculate_sustainability)

# ----------------------------
# CLEAN FINAL DATASET
# ----------------------------

final_df = df[[
    "ProductID",
    "ProductName",
    "category",
    "occasion",
    "body_type_fit",
    "PrimaryColor",
    "Price (INR)",
    "trend_score",
    "sustainability_score"
]].copy()

# Rename for consistency
final_df.columns = [
    "item_id",
    "product_name",
    "category",
    "occasion",
    "body_type_fit",
    "color",
    "price",
    "trend_score",
    "sustainability_score"
]

# Remove invalid rows
final_df = final_df[final_df["category"] != "other"]

# Save processed dataset
final_df.to_csv("processed_fashion_data.csv", index=False)

print("✅ Processed dataset created successfully!")