from services.hybrid_retriever import hybrid_search
from stylist import stylist_agent


def generate_recommendation(
    body_type: str,
    occasion: str,
    budget: float,
    sustainability: int,
    target_market: str = "men"
):
    user_profile = {
        "body_type": body_type,
        "occasion": occasion,
        "budget": budget,
        "sustainability": sustainability,
        "target_market": target_market
    }

    query = (
        f"{target_market} "
        f"{occasion} "
        f"{body_type} "
        f"fashion outfit"
    )

    results = hybrid_search(
        query=query,
        body_type=body_type,
        occasion=occasion,
        budget=budget,
        min_sustainability=sustainability,
        target_market=target_market,
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
        "products": products,
        "recommendation": recommendation
    }