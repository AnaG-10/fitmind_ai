def build_product_text(product):
    """
    Build a semantic representation of a fashion product.
    """

    parts = [
        product.get("product_name", ""),
        f"category {product.get('category', '')}",
        f"occasion {product.get('occasion', '')}",
        f"target market {product.get('target_market', '')}",
        f"color {product.get('color', '')}",
        f"fit {product.get('fit', '')}",
        f"pattern {product.get('pattern', '')}",
        f"material {product.get('material', '')}",
        f"body type {product.get('body_type_fit', '')}",
    ]

    return " ".join(
        str(part)
        for part in parts
        if part and str(part).strip()
    )