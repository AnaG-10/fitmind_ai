from fastapi import FastAPI
from pydantic import BaseModel

from services.recommendation_service import generate_recommendation


app = FastAPI(
    title="FitMind AI",
    description="AI-powered fashion recommendation API",
    version="1.0.0"
)


class RecommendationRequest(BaseModel):
    body_type: str
    occasion: str
    budget: float
    sustainability: int

    target_market: str = "men"
    category: str | None = None
    color: str | None = None
    fit: str | None = None
    material: str | None = None
    style: str | None = None


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