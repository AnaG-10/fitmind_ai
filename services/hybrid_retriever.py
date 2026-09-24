from database import get_connection
from services.qdrant_search import qdrant, COLLECTION_NAME
from services.semantic_retriever import embed_query



def get_filtered_product_ids(
    body_type,
    occasion,
    budget,
    min_sustainability=0,
    target_market="men",
    category=None,
    color=None,
    fit=None
):
    conn = get_connection()

    query = """
        SELECT item_id
        FROM products
        WHERE audience = 'adult'
          AND occasion = %s
          AND price <= %s
          AND sustainability_score >= %s
          AND (body_type_fit = %s OR body_type_fit = 'all')
          AND (target_market = %s OR target_market = 'unisex')
          AND (%s IS NULL OR category = %s)
          AND (%s IS NULL OR LOWER(color) = LOWER(%s))
          AND (%s IS NULL OR fit = %s)
    """

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                query,
                (
                    occasion,
                    budget,
                    min_sustainability,
                    body_type,
                    target_market,
                    category,
                    category,
                    color,
                    color,
                    fit,
                    fit
                )
            )

            return [row[0] for row in cursor.fetchall()]

    finally:
        conn.close()

def deduplicate_results(results):
    unique_products = {}

    for result in results:
        product = result.payload
        product_name = product["product_name"]

        if product_name not in unique_products:
            unique_products[product_name] = result
            continue

        existing = unique_products[product_name]

        existing_price = existing.payload.get("price")
        current_price = product.get("price")

        if (
            current_price is not None
            and (
                existing_price is None
                or current_price < existing_price
            )
        ):
            unique_products[product_name] = result

    return list(unique_products.values())


def hybrid_search(
    query,
    body_type,
    occasion,
    budget,
    min_sustainability=0,
    target_market="men",
    category=None,
    color=None,
    fit=None,
    limit=10
):
    filtered_ids = get_filtered_product_ids(
        body_type=body_type,
        occasion=occasion,
        budget=budget,
        min_sustainability=min_sustainability,
        target_market=target_market,
        category=category,
        color=color,
        fit=fit
    )

    if not filtered_ids:
        return []

    query_embedding = embed_query(query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        query_filter={
            "must": [
                {
                    "has_id": filtered_ids
                }
            ]
        },
        limit=limit * 3,

        with_payload=True
    )

    return deduplicate_results(results.points)[:limit]


if __name__ == "__main__":
    query = "men formal blue slim fit shirt"

    results = hybrid_search(
        query=query,
        body_type="rectangle",
        occasion="formal",
        budget=2400,
        min_sustainability=3,
        target_market="men",
        limit=5
    )

    print("\nHYBRID SEARCH RESULTS:\n")

    for i, result in enumerate(results, 1):
        product = result.payload

        print(
            f"{i}. {product['product_name']}"
            f" | {product['category']}"
            f" | {product['color']}"
            f" | ₹{product['price']}"
            f" | similarity: {result.score:.4f}"
        )
