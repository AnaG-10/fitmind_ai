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
        sustainability=request.sustainability
    )