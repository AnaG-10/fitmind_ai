
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