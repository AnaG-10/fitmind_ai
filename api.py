from fastapi.middleware.cors import CORSMiddleware
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

from services.recommendation_service import generate_recommendation


app = FastAPI(
    title="FitMind AI",
    description="AI-powered fashion recommendation API",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BodyType = Literal[
    "inverted_triangle",
    "pear",
    "rectangle"
]

Occasion = Literal[
    "casual",
    "formal",
    "party"
]

TargetMarket = Literal[
    "men",
    "women",
    "unisex"
]

Category = Literal[
    "accessory",
    "bottom",
    "footwear",
    "one_piece",
    "top"
]

FitType = Literal[
    "oversized",
    "regular",
    "relaxed",
    "skinny",
    "slim",
    "straight",
    "tailored",
    "unknown"
]


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True
    )

    body_type: BodyType
    occasion: Occasion

    budget: float = Field(
        gt=0,
        allow_inf_nan=False
    )

    sustainability: int = Field(
        ge=0,
        le=10
    )

    target_market: TargetMarket = "men"

    category: Category | None = None
    color: str | None = Field(
        default=None,
        min_length=1,
        max_length=50
    )

    fit: FitType | None = None

    material: str | None = Field(
        default=None,
        min_length=1,
        max_length=50
    )

    style: str | None = Field(
        default=None,
        min_length=1,
        max_length=50
    )


@app.get("/")
def home():
    return {
        "message": "Welcome to FitMind AI",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    return generate_recommendation(
        body_type=request.body_type,
        occasion=request.occasion,
        budget=request.budget,
        sustainability=request.sustainability,
        target_market=request.target_market,
        category=request.category,
        color=request.color,
        fit=request.fit,
        material=request.material,
        style=request.style
    )