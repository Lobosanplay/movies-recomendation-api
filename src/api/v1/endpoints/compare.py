from fastapi import APIRouter, Query, HTTPException
from services.model_service import recommendation_service
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter()

@router.get("/compare")
async def compare_movies(
    movie1: str = Query(..., description="Primera película"),
    movie2: str = Query(..., description="Segunda película")
):
    """Compara la similitud entre dos películas."""
    try:
        if not recommendation_service.is_ready():
            raise HTTPException(status_code=503, detail="Servicio no disponible")
        
        idx1 = recommendation_service.find_movie_index(movie1)
        idx2 = recommendation_service.find_movie_index(movie2)
        
        if idx1 is None or idx2 is None:
            not_found = []
            if idx1 is None:
                not_found.append(movie1)
            if idx2 is None:
                not_found.append(movie2)
            raise HTTPException(
                status_code=404, 
                detail=f"Películas no encontradas: {', '.join(not_found)}"
            )
        
        similarity = float(cosine_similarity(
            [recommendation_service.movie_vectors[idx1]],
            [recommendation_service.movie_vectors[idx2]]
        )[0][0])
        
        return {
            "movie1": recommendation_service.movie_metadata.iloc[idx1]['title'],
            "movie2": recommendation_service.movie_metadata.iloc[idx2]['title'],
            "similarity_score": similarity,
            "interpretation": {
                "0-0.3": "Poca similitud",
                "0.3-0.6": "Moderadamente similares",
                "0.6-0.8": "Muy similares",
                "0.8-1.0": "Extremadamente similares"
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")