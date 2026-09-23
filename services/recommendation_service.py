from recommender import filter_items
from stylist import stylist_agent


def generate_recommendation(
    body_type: str,
    occasion: str,
    budget: float,
    sustainability: int
):

    user_profile = {
        "body_type": body_type,
        "occasion": occasion,
        "budget": budget,
        "sustainability": sustainability
    }

    items = filter_items(
        body_type,
        occasion,
        budget,
        sustainability
    )

    if not items:
        return {
            "success": False,
            "message": "No suitable products found.",
            "user_profile": user_profile,
            "products": []
        }

    recommendation = stylist_agent(
        user_profile,
        items
    )

    return {
        "success": True,
        "user_profile": user_profile,
        "products": items,
        "recommendation": recommendation
    }