from database import get_connection
from services.hybrid_retriever import hybrid_search
from stylist import stylist_agent


def build_semantic_query(
    body_type,
    occasion,
    target_market,
    category=None,
    color=None,
    fit=None,
    material=None,
    style=None
):
    query_parts = [
        target_market,
        occasion,
        body_type
    ]

    preferences = {
        "category": category,
        "color": color,
        "fit": fit,
        "material": material,
        "style": style
    }

    for label, value in preferences.items():
        if value and value.strip():
            query_parts.append(f"{label} {value.strip()}")

    return " ".join(query_parts)

def enrich_products_from_db(products):
    if not products:
        return products

    item_ids = [product["item_id"] for product in products]

    conn = get_connection()

    query = """
        SELECT
            item_id,
            product_name,
            brand,
            description,
            category,
            occasion,
            color,
            price,
            trend_score,
            sustainability_score,
            target_market,
            fit,
            pattern,
            material
        FROM products
        WHERE item_id = ANY(%s)
    """

    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (item_ids,))

            rows = cursor.fetchall()

            columns = [desc[0] for desc in cursor.description]

            db_products = {
                row[0]: dict(zip(columns, row))
                for row in rows
            }

        enriched = []

        for product in products:
            item_id = product["item_id"]

            db_product = db_products.get(item_id, {})

            enriched_product = {
                **db_product,
                **product,
                "brand": db_product.get("brand"),
                "description": db_product.get("description")
            }

            enriched.append(enriched_product)

        return enriched

    finally:
        conn.close()
        
def generate_recommendation(
    body_type: str,
    occasion: str,
    budget: float,
    sustainability: int,
    target_market: str = "men",
    category: str = None,
    color: str = None,
    fit: str = None,
    material: str = None,
    style: str = None
):
    user_profile = {
        "body_type": body_type,
        "occasion": occasion,
        "budget": budget,
        "sustainability": sustainability,
        "target_market": target_market,
        "category": category,
        "color": color,
        "fit": fit,
        "material": material,
        "style": style
    }

    query = build_semantic_query(
        body_type=body_type,
        occasion=occasion,
        target_market=target_market,
        category=category,
        color=color,
        fit=fit,
        material=material,
        style=style
    )

    results = hybrid_search(
    query=query,
    body_type=body_type,
    occasion=occasion,
    budget=budget,
    min_sustainability=sustainability,
    target_market=target_market,
    category=category,
    color=color,
    fit=fit,
    limit=10
)

    products = []

    for result in results:
        product = dict(result.payload)
        product["semantic_score"] = round(float(result.score), 4)
        products.append(product)

    products = enrich_products_from_db(products)

    if not products:
        return {
            "success": False,
            "message": "No suitable products found.",
            "user_profile": user_profile,
            "products": []
        }

    recommendation = stylist_agent(
        user_profile,
        products
    )

    return {
        "success": True,
        "user_profile": user_profile,
        "semantic_query": query,
        "products": products,
        "recommendation": recommendation
    }