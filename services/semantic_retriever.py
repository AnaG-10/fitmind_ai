from sentence_transformers import SentenceTransformer

from services.embedding_text import build_product_text


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def embed_product(product):
    """
    Convert a product dictionary into a semantic embedding.
    """

    text = build_product_text(product)

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding


def embed_query(query):
    """
    Convert a user fashion query into a semantic embedding.
    """

    embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    return embedding