from database import get_connection
from services.embedding_text import build_product_text
from services.semantic_retriever import embed_query

from sentence_transformers import util


def load_products(limit=100):
    conn = get_connection()

    query = """
        SELECT
            item_id,
            product_name,
            category,
            occasion,
            target_market,
            color,
            fit,
            pattern,
            material,
            body_type_fit
        FROM products
        WHERE audience = 'adult'
        LIMIT %s;
    """

    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (limit,))

            columns = [description[0] for description in cursor.description]

            return [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]

    finally:
        conn.close()


def semantic_search(query, limit=5):
    products = load_products(200)

    query_embedding = embed_query(query)

    product_texts = [
        build_product_text(product)
        for product in products
    ]

    from services.semantic_retriever import model

    product_embeddings = model.encode(
        product_texts,
        normalize_embeddings=True
    )

    similarities = util.cos_sim(
        query_embedding,
        product_embeddings
    )[0]

    ranked = sorted(
        zip(products, similarities),
        key=lambda x: float(x[1]),
        reverse=True
    )

    return ranked[:limit]


if __name__ == "__main__":

    query = "men formal blue slim fit shirt"

    results = semantic_search(query)

    print("\nQUERY:", query)
    print("\nTOP SEMANTIC MATCHES:\n")

    for i, (product, score) in enumerate(results, 1):

        print(
            f"{i}.",
            product["product_name"],
            "|",
            product["category"],
            "|",
            product["color"],
            "|",
            product["fit"],
            "| similarity:",
            round(float(score), 4)
        )