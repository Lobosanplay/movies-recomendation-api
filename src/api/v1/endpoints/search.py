from fastapi import APIRouter, Query, HTTPException
from services.model_service import recommendation_service


router = APIRouter()

@router.get("/search/movies")
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

@router.get("/search/genres")
async def search_genres(
    query: str = Query(None, description="Texto para buscar géneros (opcional)"),
    match_type: str = Query("starts_with", description="Tipo de coincidencia: 'starts_with' o 'contains'")
):
    """Obtiene todos los géneros únicos disponibles, filtrados por query si se proporciona."""
    if not recommendation_service.is_ready():
        raise HTTPException(status_code=503, detail="Servicio no disponible")
    
    if recommendation_service.movie_metadata is not None:
        genres_series = recommendation_service.movie_metadata['genres']
        
        all_genres = set()
        
        if isinstance(genres_series.iloc[0], str):
            for genre_str in genres_series.dropna():
                genres = [g.strip() for g in genre_str.split(',')]
                all_genres.update(genres)
        
        elif isinstance(genres_series.iloc[0], list):
            for genre_list in genres_series.dropna():
                all_genres.update(genre_list)
        
        sorted_genres = sorted(list(all_genres))
        
        filtered_genres = []
        if query:
            query_lower = query.lower()
            for genre in sorted_genres:
                genre_lower = genre.lower()
                
                if match_type == "starts_with":
                    if genre_lower.startswith(query_lower):
                        filtered_genres.append(genre)
                elif match_type == "contains":
                    if query_lower in genre_lower:
                        filtered_genres.append(genre)
                else:
                    if genre_lower.startswith(query_lower):
                        filtered_genres.append(genre)
        else:
            filtered_genres = sorted_genres
        
        return {
            "total_genres": len(filtered_genres),
            "genres": filtered_genres,
            "original_total": len(sorted_genres),
            "query_used": query,
            "match_type": match_type
        }
    
    return {"total_genres": 0, "genres": []}