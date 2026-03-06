import pandas as pd

def filter_items(body_type, occasion, budget, min_sustainability=0):
    df = pd.read_csv("processed_fashion_data.csv")

    df = df[
        ((df["body_type_fit"] == body_type) | (df["body_type_fit"] == "all")) &
        (df["occasion"] == occasion) &
        (df["price"] <= budget) &
        (df["sustainability_score"] >= min_sustainability)
    ]

    df = df.sort_values(
        by=["trend_score", "sustainability_score"],
        ascending=False
    )

    return df.head(10).to_dict(orient="records")