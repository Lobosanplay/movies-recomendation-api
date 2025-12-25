from fastapi import FastAPI
from services.model_service import recommendation_service
from api.v1.router import api_router

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Se ejecuta al iniciar la aplicación"""
    await recommendation_service.initialize()

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health"
    }