
from unittest.mock import patch
from fastapi.testclient import TestClient

import api


client = TestClient(api.app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_invalid_body_type():
    response = client.post(
        "/recommend",
        json={
            "body_type": "apple",
            "occasion": "formal",
            "budget": 2400,
            "sustainability": 3,
            "target_market": "men"
        }
    )

    assert response.status_code == 422


def test_invalid_category():
    response = client.post(
        "/recommend",
        json={
            "body_type": "rectangle",
            "occasion": "formal",
            "budget": 2400,
            "sustainability": 3,
            "target_market": "men",
            "category": "clothing"
        }
    )

    assert response.status_code == 422


def test_invalid_fit():
    response = client.post(
        "/recommend",
        json={
            "body_type": "rectangle",
            "occasion": "formal",
            "budget": 2400,
            "sustainability": 3,
            "target_market": "men",
            "fit": "baggy"
        }
    )

    assert response.status_code == 422


def test_invalid_target_market():
    response = client.post(
        "/recommend",
        json={
            "body_type": "rectangle",
            "occasion": "formal",
            "budget": 2400,
            "sustainability": 3,
            "target_market": "children"
        }
    )

    assert response.status_code == 422


def test_invalid_occasion():
    response = client.post(
        "/recommend",
        json={
            "body_type": "rectangle",
            "occasion": "wedding",
            "budget": 2400,
            "sustainability": 3,
            "target_market": "men"
        }
    )

    assert response.status_code == 422


def test_invalid_budget():
    response = client.post(
        "/recommend",
        json={
            "body_type": "rectangle",
            "occasion": "formal",
            "budget": 0,
            "sustainability": 3,
            "target_market": "men"
        }
    )

    assert response.status_code == 422


def test_invalid_sustainability():
    response = client.post(
        "/recommend",
        json={
            "body_type": "rectangle",
            "occasion": "formal",
            "budget": 2400,
            "sustainability": 11,
            "target_market": "men"
        }
    )

    assert response.status_code == 422
    

def test_valid_recommendation():
    mock_result = {
        "success": True,
        "products": [
            {
                "product_name": "Test Formal Shirt",
                "price": 599,
                "category": "top"
            }
        ],
        "recommendation": "Test stylist recommendation"
    }

    with patch(
        "api.generate_recommendation",
        return_value=mock_result
    ) as mock_recommendation:

        response = client.post(
            "/recommend",
            json={
                "body_type": "rectangle",
                "occasion": "formal",
                "budget": 2400,
                "sustainability": 3,
                "target_market": "men",
                "category": "top",
                "color": "White",
                "fit": "regular"
            }
        )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["products"][0]["product_name"] == "Test Formal Shirt"
    assert response.json()["recommendation"] == "Test stylist recommendation"

    mock_recommendation.assert_called_once()    