from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ReviewRequest(BaseModel):
    paper_title: str
    reviews: list

@app.post("/generate_review")
async def generate_review(request: ReviewRequest):
    """生成统一格式的review"""
    aggregated = review_aggregator.aggregate_reviews(request.reviews)
    return {"review": aggregated}