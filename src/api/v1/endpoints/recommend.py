from fastapi import APIRouter, Query, HTTPException
from services.model_service import recommendation_service
from schemas.recomment_schemas import TagsRequest

router = APIRouter()

@router.get("/by-title")
async def recommend_by_title(
    title: str = Query(..., description="Título de la película"),
    limit: int = Query(5, ge=1, le=20, description="Número de recomendaciones")
):
    """
    Obtiene recomendaciones basadas en una película.
    
    Ejemplo: /recommend/by-title?title=Inception&limit=5
    """
    try:
        recommendations = recommendation_service.get_similar_movies(title, top_n=limit)
        
        return {
            "query": title,
            "limit": limit,
            "found_movies": len(recommendations),
            "recommendations": recommendations
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
    
@router.post("/by-tags")
async def recommend_by_tags(request: TagsRequest):
    """
    Obtiene recomendaciones basadas en tags.
    
    Ejemplo de request body:
    {
        "tags": ["sci-fi", "space", "adventure"],
        "limit": 5
    }
    """
    try:
        recommendations = recommendation_service.recommend_by_tags(
            request.tags, 
            top_n=request.limit
        )
        
        return {
            "query_tags": request.tags,
            "limit": request.limit,
            "found_movies": len(recommendations),
            "recommendations": recommendations
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")