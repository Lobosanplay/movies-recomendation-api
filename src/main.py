from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.v1.router import api_router
from services.model_service import recommendation_service


@asynccontextmanager
async def startup_event(app: FastAPI):
    """Se ejecuta al iniciar la aplicación"""
    await recommendation_service.initialize()
    yield


app = FastAPI(lifespan=startup_event)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=True)
