from database import get_connection


def get_category_candidates(
    body_type,
    occasion,
    budget,
    min_sustainability=0,
    target_market="men"
):
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
        WHERE
            audience = 'adult'
            AND occasion = %s
            AND price <= %s
            AND sustainability_score >= %s
            AND (body_type_fit = %s OR body_type_fit = 'all')
            AND (target_market = %s OR target_market = 'unisex')
            AND category IN ('top', 'bottom', 'footwear', 'accessory')
            
        ORDER BY
            CASE
                WHEN body_type_fit = %s THEN 0
                ELSE 1
            END,
            trend_score DESC,
            sustainability_score DESC,
            price ASC;
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
                    body_type,
                )
            )

            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

            candidates = [dict(zip(columns, row)) for row in rows]

            grouped = {
                "top": [],
                "bottom": [],
                "footwear": [],
                "accessory": []
            }

            for product in candidates:
                grouped[product["category"]].append(product)

            return grouped

    finally:
        conn.close()
        
def diversify_candidates(products, limit):
    """
    Select a diverse candidate pool based on
    color, fit, and pattern.
    """

    selected = []
    seen_signatures = set()

    for product in products:

        signature = (
            product.get("color"),
            product.get("fit"),
            product.get("pattern")
        )

        if signature in seen_signatures:
            continue

        selected.append(product)
        seen_signatures.add(signature)

        if len(selected) >= limit:
            break

    # Fill remaining slots if necessary
    if len(selected) < limit:

        selected_ids = {
            product["item_id"]
            for product in selected
        }

        for product in products:

            if product["item_id"] in selected_ids:
                continue

            selected.append(product)

            if len(selected) >= limit:
                break

    return selected        

COLOR_GROUPS = {
    "white": "neutral",
    "grey": "neutral",
    "black": "neutral",
    "beige": "neutral",
    "brown": "neutral",
    "khaki": "neutral",

    "blue": "cool",
    "green": "cool",
    "purple": "cool",
    "lavender": "cool",

    "pink": "warm",
    "red": "warm",
    "orange": "warm",
    "maroon": "warm",
    "burgundy": "warm",
    "magenta": "warm",
    "yellow": "warm",
}


def get_color_group(color):
    if not color:
        return "unknown"

    return COLOR_GROUPS.get(
        color.strip().lower(),
        "unknown"
    )


def calculate_color_compatibility(products):
    """
    Score color compatibility between the main outfit pieces.

    Returns a score out of 20.
    """

    colors = [
        product.get("color")
        for product in products
        if product.get("category") in ("top", "bottom", "footwear")
    ]

    groups = [
        get_color_group(color)
        for color in colors
    ]

    groups = [group for group in groups if group != "unknown"]

    if len(groups) < 2:
        return 12

    score = 20

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):

            first = groups[i]
            second = groups[j]

            if first == "neutral" or second == "neutral":
                score += 0

            elif first == second:
                score -= 1

            else:
                score -= 4

    return max(0, min(score, 20))
def calculate_style_coherence(products):
    """
    Score fit and pattern consistency across outfit pieces.
    Returns a score out of 10.
    """

    main_products = [
        product
        for product in products
        if product["category"] in ("top", "bottom", "footwear")
    ]

    score = 10

    # -----------------------------
    # Fit consistency
    # -----------------------------

    fits = [
        product["fit"]
        for product in main_products
        if product["fit"] != "unknown"
    ]

    if len(fits) >= 2:
        if len(set(fits)) == 1:
            score += 0
        elif "slim" in fits and "regular" in fits:
            score -= 1
        elif "skinny" in fits and "regular" in fits:
            score -= 2

    # -----------------------------
    # Pattern compatibility
    # -----------------------------

    patterns = [
        product["pattern"]
        for product in main_products
        if product["pattern"] != "unknown"
    ]

    if len(patterns) >= 2:

        patterned = [
            pattern
            for pattern in patterns
            if pattern != "solid"
        ]

        # Multiple strong patterns can make formal outfits busy.
        if len(patterned) >= 2:
            score -= 2

        # Solid + one pattern is generally safer.
        elif len(patterned) == 1:
            score += 0

    return max(0, min(score, 10))
def calculate_outfit_score(outfit, budget):
    """
    Transparent outfit-level score out of 100.
    """

    products = outfit["products"]

    # -----------------------------
    # 1. Product quality: 30
    # -----------------------------

    product_quality = 0

    for product in products:

        trend = product["trend_score"]
        sustainability = product["sustainability_score"]

        product_score = (
            (trend / 10) * 5
            +
            (sustainability / 10) * 5
        )

        # Reward products with explicitly detected attributes
        if product["fit"] != "unknown":
            product_score += 1

        if product["pattern"] != "unknown":
            product_score += 1

        if product["material"] != "unknown":
            product_score += 1

        product_quality += product_score

    product_quality = min(product_quality, 30)

    # -----------------------------
    # 2. Body-type compatibility: 20
    # -----------------------------

    body_specific = sum(
        1
        for product in products
        if product["body_type_fit"] != "all"
    )

    if body_specific == len(products):
        body_score = 20
    elif body_specific > 0:
        body_score = 15
    else:
        body_score = 10

    # -----------------------------
    # 3. Color compatibility: 20
    # -----------------------------

    color_score = calculate_color_compatibility(products)

    # -----------------------------
    # 4. Occasion consistency: 15
    # -----------------------------

    occasions = [
        product["occasion"]
        for product in products
    ]

    occasion_score = (
        15
        if len(set(occasions)) == 1
        else 5
    )

    # -----------------------------
    # 5. Style coherence: 10
    # -----------------------------

    style_score = calculate_style_coherence(products)

    # -----------------------------
    # 6. Budget efficiency: 5
    # -----------------------------

    total_price = outfit["total_price"]

    budget_usage = (
        total_price / budget
        if budget > 0
        else 0
    )

    budget_score = min(budget_usage, 1) * 5

    # -----------------------------
    # Final score
    # -----------------------------

    total_score = (
        product_quality
        + body_score
        + color_score
        + occasion_score
        + style_score
        + budget_score
    )

    return round(total_score, 2)
def are_outfits_similar(outfit_a, outfit_b):
    """
    Determine whether two outfits are too similar
    to be shown as separate recommendations.
    """

    def get_signature(outfit):

        products = {
            product["category"]: product
            for product in outfit["products"]
        }

        top = products.get("top", {})
        bottom = products.get("bottom", {})
        footwear = products.get("footwear", {})

        return {
            "top_color": get_color_group(top.get("color")),
            "bottom_color": get_color_group(bottom.get("color")),
            "shoe_color": get_color_group(footwear.get("color")),
            "top_pattern": top.get("pattern"),
            "top_fit": top.get("fit"),
            "bottom_fit": bottom.get("fit"),
        }

    a = get_signature(outfit_a)
    b = get_signature(outfit_b)

    differences = 0

    if a["top_color"] != b["top_color"]:
        differences += 1

    if a["bottom_color"] != b["bottom_color"]:
        differences += 1

    if a["shoe_color"] != b["shoe_color"]:
        differences += 1

    if a["top_pattern"] != b["top_pattern"]:
        differences += 1

    if a["top_fit"] != b["top_fit"]:
        differences += 1

    if a["bottom_fit"] != b["bottom_fit"]:
        differences += 1

    # If fewer than 2 major style dimensions change,
    # treat them as the same outfit concept.
    return differences < 2

   

def generate_outfits(
    body_type,
    occasion,
    budget,
    min_sustainability=0,
    max_outfits=5,
    target_market="men"
):
    """
    Generate complete outfits consisting of:

        Top + Bottom + Footwear

    Accessories are optional.

    Every generated outfit must remain within the user's budget.
    """

    candidates = get_category_candidates(
        body_type,
        occasion,
        budget,
        min_sustainability,
        target_market
    )

    tops = diversify_candidates(candidates["top"], 50)
    bottoms = diversify_candidates(candidates["bottom"], 30)
    footwear = diversify_candidates(candidates["footwear"], 20)
    accessories = diversify_candidates(candidates["accessory"], 10)
    outfits = []

    for top in tops:
        for bottom in bottoms:
            for shoe in footwear:

                products = [
                    top,
                    bottom,
                    shoe
                ]

                total_price = sum(
                    float(product["price"])
                    for product in products
                )

                if total_price > budget:
                    continue

                outfit = {
                    "products": products,
                    "total_price": total_price,
                    "has_accessory": False
                }

                outfit["outfit_score"] = calculate_outfit_score(outfit,budget)
                outfits.append(outfit)

                # Try adding an accessory if one fits
                for accessory in accessories:

                    accessory_total = (
                        total_price
                        + float(accessory["price"])
                    )

                    if accessory_total <= budget:

                        accessory_outfit = {
                            "products": [
                                top,
                                bottom,
                                shoe,
                                accessory
                            ],
                            "total_price": accessory_total,
                            "has_accessory": True
                        }

                        accessory_outfit["outfit_score"] = (
                            calculate_outfit_score(accessory_outfit,budget)
)

                        outfits.append(accessory_outfit)

    outfits.sort(
        key=lambda outfit: (
            outfit["outfit_score"],
            -outfit["total_price"]
        ),
        reverse=True
    )

    selected_outfits = []

    for outfit in outfits:
        if all(
            not are_outfits_similar(outfit, selected)
            for selected in selected_outfits
        ):
            selected_outfits.append(outfit)

        if len(selected_outfits) == max_outfits:
            break

    return selected_outfits