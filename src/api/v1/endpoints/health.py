from fastapi import APIRouter
from services.model_service import recommendation_service

router = APIRouter()

@router.get("")
async def health_check():
    """Verifica el estado del servicio."""
    stats = recommendation_service.get_stats()
    
    return {
        "status": "healthy" if recommendation_service.is_ready() else "initializing",
        "service_stats": stats,
        "message": "Recommendation service is ready" 
        if recommendation_service.is_ready() 
        else "Service is initializing, please wait"
    }

@router.get("/ready")
async def readiness_probe():
    """Endpoint para verificar si el servicio está listo."""
    if recommendation_service.is_ready():
        return {"status": "ready"}
    return {"status": "not_ready"}, 503
