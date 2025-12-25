from fastapi import APIRouter
from api.v1.endpoints import recommend, health

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(recommend.router, prefix="/recommend", tags=["recommend"])
