from fastapi import APIRouter, Query, HTTPException
from services.model_service import recommendation_service

router = APIRouter()

@router.get("/search")
async def search_movies(
    query: str = Query(..., description="Texto para buscar"),
    limit: int = Query(10, ge=1, le=50, description="Resultados máximos")
):
    """Busca películas por título."""
    if not recommendation_service.is_ready():
        raise HTTPException(status_code=503, detail="Servicio no disponible")
    
    if recommendation_service.movie_metadata is not None:
        filtered = recommendation_service.movie_metadata[
            recommendation_service.movie_metadata['title']
            .str.contains(query, case=False, na=False)
        ].head(limit)
        
        return {
            "query": query,
            "limit": limit,
            "found": len(filtered),
            "movies": filtered.to_dict('records')
        }
    
    return {"query": query, "found": 0, "movies": []}