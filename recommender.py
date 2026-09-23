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
            sustainability_score
        FROM products
        WHERE (body_type_fit = %s OR body_type_fit = 'all')
          AND occasion = %s
          AND price <= %s
          AND sustainability_score >= %s
        ORDER BY trend_score DESC, sustainability_score DESC
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