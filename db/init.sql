CREATE TABLE IF NOT EXISTS products (
    item_id BIGINT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT,
    occasion TEXT,
    body_type_fit TEXT,
    color TEXT,
    price NUMERIC(10,2),
    trend_score INTEGER,
    sustainability_score INTEGER
);

CREATE INDEX IF NOT EXISTS idx_products_occasion
ON products (occasion);

CREATE INDEX IF NOT EXISTS idx_products_body_type
ON products (body_type_fit);

CREATE INDEX IF NOT EXISTS idx_products_price
ON products (price);

CREATE INDEX IF NOT EXISTS idx_products_sustainability
ON products (sustainability_score);

CREATE INDEX IF NOT EXISTS idx_products_trend
ON products (trend_score);