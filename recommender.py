from database import get_connection


def filter_items(
    body_type,
    occasion,
    budget,
    min_sustainability=0
):
    conn = get_connection()

    query = """
        WITH ranked_products AS (
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

                (
                    CASE
                        WHEN body_type_fit = %s THEN 40
                        WHEN body_type_fit = 'all' THEN 25
                        ELSE 0
                    END

                    +

                    CASE
                        WHEN occasion = %s THEN 25
                        ELSE 0
                    END

                    +

                    CASE
                        WHEN price <= %s * 0.25 THEN 15
                        WHEN price <= %s * 0.50 THEN 13
                        WHEN price <= %s * 0.75 THEN 11
                        WHEN price <= %s THEN 9
                        ELSE 0
                    END

                    +

                    (sustainability_score * 1.0)

                    +

                    (trend_score * 0.5)

                    +

                    CASE
                        WHEN product_name ILIKE '%%slim fit%%' THEN 2
                        WHEN product_name ILIKE '%%regular fit%%' THEN 1
                        ELSE 0
                    END

                    +

                    CASE
                        WHEN product_name ILIKE '%%linen%%' THEN 2
                        WHEN product_name ILIKE '%%cotton%%' THEN 1
                        ELSE 0
                    END

                    +

                    CASE
                        WHEN product_name ILIKE '%%solid%%' THEN 1
                        ELSE 0
                    END

                ) AS match_score

            FROM products

            WHERE (body_type_fit = %s OR body_type_fit = 'all')
            AND occasion = %s
            AND price <= %s
            AND sustainability_score >= %s
            AND audience = 'adult'
        ),

        category_ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY category
                    ORDER BY match_score DESC, price ASC
                ) AS category_rank
            FROM ranked_products
        )

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
            match_score

        FROM category_ranked

        WHERE
            (category = 'top' AND category_rank <= 4)
            OR
            (category = 'bottom' AND category_rank <= 3)
            OR
            (category = 'footwear' AND category_rank <= 2)
            OR
            (category = 'accessory' AND category_rank <= 1)

        ORDER BY match_score DESC, price ASC;
    """

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                query,
                (
                    body_type,
                    occasion,
                    budget,
                    budget,
                    budget,
                    budget,
                    body_type,
                    occasion,
                    budget,
                    min_sustainability
                )
            )

            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

            return [
                dict(zip(columns, row))
                for row in rows
            ]

    finally:
        conn.close()