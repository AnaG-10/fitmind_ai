from database import get_connection


def filter_items(
    body_type,
    occasion,
    budget,
    min_sustainability=0
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
                    WHEN price <= %s * 0.5 THEN 15
                    WHEN price <= %s * 0.75 THEN 12
                    WHEN price <= %s THEN 10
                    ELSE 0
                END
                +
                (sustainability_score * 1.0)
                +
                (trend_score * 0.5)
            ) AS match_score

        FROM products

        WHERE (body_type_fit = %s OR body_type_fit = 'all')
          AND occasion = %s
          AND price <= %s
          AND sustainability_score >= %s

        ORDER BY match_score DESC, price ASC

        LIMIT 10;
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