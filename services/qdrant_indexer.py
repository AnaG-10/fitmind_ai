from database import get_connection
from services.embedding_text import build_product_text
from services.semantic_retriever import model

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "fashion_products"

BATCH_SIZE = 100

qdrant = QdrantClient(url=QDRANT_URL)


def load_products(offset, limit):
    conn = get_connection()

    query = """
        SELECT
            item_id,
            product_name,
            category,
            occasion,
            body_type_fit,
            color,
            price,
            trend_score,
            sustainability_score,
            audience,
            target_market,
            fit,
            pattern,
            material
        FROM products
        WHERE audience = 'adult'
        ORDER BY item_id
        LIMIT %s OFFSET %s
    """

    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (limit, offset))
            columns = [description[0] for description in cursor.description]

            return [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]
    finally:
        conn.close()


def create_embeddings(products):
    texts = [
        build_product_text(product)
        for product in products
    ]

    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )


def upload_products(products, embeddings):
    points = []

    for product, embedding in zip(products, embeddings):

        payload = {
            "item_id": product["item_id"],
            "product_name": product["product_name"],
            "category": product["category"],
            "occasion": product["occasion"],
            "body_type_fit": product["body_type_fit"],
            "color": product["color"],
            "price": (
                float(product["price"])
                if product["price"] is not None
                else None
            ),
            "trend_score": product["trend_score"],
            "sustainability_score": product["sustainability_score"],
            "audience": product["audience"],
            "target_market": product["target_market"],
            "fit": product["fit"],
            "pattern": product["pattern"],
            "material": product["material"],
        }

        points.append(
            PointStruct(
                id=int(product["item_id"]),
                vector=embedding.tolist(),
                payload=payload
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )


def index_catalog():
    offset = 0
    total_indexed = 0

    while True:
        products = load_products(
            offset=offset,
            limit=BATCH_SIZE
        )

        if not products:
            break

        print(
            f"Processing products "
            f"{offset + 1}-{offset + len(products)}"
        )

        embeddings = create_embeddings(products)

        upload_products(
            products,
            embeddings
        )

        total_indexed += len(products)
        offset += BATCH_SIZE

        print(
            f"Uploaded: {total_indexed} products"
        )

    print()
    print("================================")
    print("Qdrant indexing completed")
    print("Total indexed:", total_indexed)
    print("================================")


if __name__ == "__main__":
    index_catalog()