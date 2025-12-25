from fastapi import APIRouter
from api.v1.endpoints import recommend, health, search, compare

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(recommend.router, prefix="/recommend", tags=["recommend movies"])
api_router.include_router(search.router, prefix="/search", tags=["search movies"])
api_router.include_router(compare.router, prefix="/compare", tags=["compare movies"])