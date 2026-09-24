import pandas as pd
from sqlalchemy import create_engine, text

# Dataset paths
PROCESSED_CSV = "processed_fashion_data.csv"
ORIGINAL_CSV = "data/myntra_products_catalog.csv"

# Update this if your existing database URL is different.
DATABASE_URL = "postgresql+psycopg2://fitmind:fitmind_password@localhost:5432/fitmind"
# Load datasets
processed = pd.read_csv(PROCESSED_CSV)
original = pd.read_csv(ORIGINAL_CSV)

# Keep only the enrichment fields
metadata = original[
    ["ProductID", "ProductBrand", "Description"]
].copy()

metadata.columns = [
    "item_id",
    "brand",
    "description",
]

# Match original Myntra metadata to processed products
enriched = processed[["item_id"]].merge(
    metadata,
    on="item_id",
    how="left",
    validate="one_to_one",
)

print("Products to enrich:", len(enriched))
print("Missing brands:", enriched["brand"].isna().sum())
print("Missing descriptions:", enriched["description"].isna().sum())

# Create database connection
engine = create_engine(DATABASE_URL)

# Update existing rows only
with engine.begin() as connection:
    for _, row in enriched.iterrows():
        connection.execute(
            text("""
                UPDATE products
                SET brand = :brand,
                    description = :description
                WHERE item_id = :item_id
            """),
            {
                "item_id": int(row["item_id"]),
                "brand": (
                    None if pd.isna(row["brand"])
                    else row["brand"]
                ),
                "description": (
                    None if pd.isna(row["description"])
                    else row["description"]
                ),
            },
        )

print("Product enrichment completed.")