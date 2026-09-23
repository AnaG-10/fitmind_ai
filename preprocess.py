import pandas as pd
import re


# Load dataset
df = pd.read_csv("data/myntra_products_catalog.csv")


# --------------------------------------------------
# COMBINE RELEVANT TEXT
# --------------------------------------------------

df["text"] = (
    df["ProductName"].fillna("") + " " +
    df["Description"].fillna("")
).str.lower()


# --------------------------------------------------
# CATEGORY DETECTION
# --------------------------------------------------

def detect_category(text):
    if re.search(
        r"\b(shirt|t[- ]?shirt|top|kurta|blouse|jacket|sweater|hoodie)\b",
        text
    ):
        return "top"

    elif re.search(
        r"\b(jeans|trouser|trousers|pants|shorts|skirt|leggings|joggers)\b",
        text
    ):
        return "bottom"

    elif re.search(
        r"\b(shoes|heels|sneakers|sandals|boots|loafers|slip-ons)\b",
        text
    ):
        return "footwear"

    elif re.search(
        r"\b(dress|gown|saree|sari|jumpsuit|romper)\b",
        text
    ):
        return "one_piece"

    elif re.search(
        r"\b(bag|handbag|backpack|wallet|belt|watch|sunglasses)\b",
        text
    ):
        return "accessory"

    else:
        return "other"


df["category"] = df["text"].apply(detect_category)


# --------------------------------------------------
# OCCASION DETECTION
# --------------------------------------------------

def detect_occasion(text):

    # Explicit sleep/nightwear should never become formal
    if re.search(
        r"\b(night suit|nightwear|sleepwear|sleep wear|pyjama|pajama|"
        r"nightdress|night dress|lounge wear|loungewear)\b",
        text
    ):
        return "casual"

    # Party/festive occasions
    if re.search(
        r"\b(party|evening|festive|wedding|cocktail|celebration)\b",
        text
    ):
        return "party"

    # Formal occasions
    if re.search(
        r"\b(formal|office wear|officewear|business wear|businesswear|"
        r"corporate|professional|workwear|work wear)\b",
        text
    ):
        return "formal"

    return "casual"


df["occasion"] = df["text"].apply(detect_occasion)


# --------------------------------------------------
# BODY TYPE FIT
# --------------------------------------------------
# This is a heuristic, not a physical measurement.
# Defaulting to "all" is preferable to random assignment.

def detect_body_fit(text):

    if re.search(
        r"\b(high[- ]waist|high[- ]rise|a[- ]line|flared|flare)\b",
        text
    ):
        return "pear"

    elif re.search(
        r"\b(oversized|relaxed fit|boxy)\b",
        text
    ):
        return "rectangle"

    elif re.search(
        r"\b(structured shoulder|broad shoulder|shoulder detail)\b",
        text
    ):
        return "inverted_triangle"

    else:
        return "all"


df["body_type_fit"] = df["text"].apply(detect_body_fit)


# --------------------------------------------------
# TREND SCORE
# --------------------------------------------------

trend_keywords = {
    "oversized": 1,
    "cropped": 1,
    "streetwear": 1,
    "street style": 1,
    "trendy": 1,
    "baggy": 1,
    "co-ord": 1,
    "co ord": 1
}


def calculate_trend_score(text):
    score = 5

    for keyword, points in trend_keywords.items():
        if keyword in text:
            score += points

    return min(score, 10)


df["trend_score"] = df["text"].apply(calculate_trend_score)


# --------------------------------------------------
# SUSTAINABILITY SCORE
# --------------------------------------------------

sustainable_keywords = {
    "organic": 2,
    "recycled": 2,
    "eco": 1,
    "sustainable": 2,
    "linen": 1,
    "bamboo": 1,
    "reusable": 1
}


def calculate_sustainability(text):
    score = 5

    for keyword, points in sustainable_keywords.items():
        if keyword in text:
            score += points

    return min(score, 10)


df["sustainability_score"] = df["text"].apply(
    calculate_sustainability
)


# --------------------------------------------------
# CLEAN FINAL DATASET
# --------------------------------------------------

final_df = df[
    [
        "ProductID",
        "ProductName",
        "category",
        "occasion",
        "body_type_fit",
        "PrimaryColor",
        "Price (INR)",
        "trend_score",
        "sustainability_score"
    ]
].copy()


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


# Remove products that could not be classified
final_df = final_df[final_df["category"] != "other"]


# Remove invalid prices
final_df = final_df[
    final_df["price"].notna() &
    (final_df["price"] > 0)
]


# Clean whitespace
final_df["color"] = final_df["color"].astype(str).str.strip()
final_df["product_name"] = final_df["product_name"].astype(str).str.strip()


# Save processed dataset
final_df.to_csv(
    "processed_fashion_data.csv",
    index=False
)


print("Processed dataset created successfully!")
print(f"Products: {len(final_df)}")
print("\nCategories:")
print(final_df["category"].value_counts())

print("\nOccasions:")
print(final_df["occasion"].value_counts())

print("\nBody type fit:")
print(final_df["body_type_fit"].value_counts())