from qdrant_client import QdrantClient
from services.semantic_retriever import embed_query


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "fashion_products"

qdrant = QdrantClient(url=QDRANT_URL)


def semantic_search(query, limit=5):
    query_embedding = embed_query(query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=limit,
        with_payload=True
    )

    return results.points


if __name__ == "__main__":
    query = "men formal blue slim fit shirt"

    results = semantic_search(query, limit=5)

    print("\nQUERY:", query)
    print("\nTOP QDRANT MATCHES:\n")

    for i, result in enumerate(results, 1):
        product = result.payload

        print(
            f"{i}. {product['product_name']} "
            f"| {product['category']} "
            f"| {product['color']} "
            f"| {product['fit']} "
            f"| similarity: {result.score:.4f}"
        )