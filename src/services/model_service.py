import os
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import HTTPException
from sklearn.metrics.pairwise import cosine_similarity

from services.model_creator_service import create_model


class RecommendationModelService:
    """Servicio Singleton para manejar el modelo de recomendación."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.movie_vectors = None
            self.vectorizer = None
            self.movie_metadata = None
            self.movie_titles_map = {}
            self.initialized = True

    async def initialize(
        self,
        model_path: str = "src/models/recomendation_model.pkl",
        create_if_missing: bool = True,
    ):
        """
        Inicializa el servicio cargando el modelo.

        Args:
            model_path: Ruta al modelo
            create_if_missing: Si es True, crea el modelo si no existe
        """
        try:
            if not os.path.exists(model_path) and create_if_missing:
                print("Modelo no encontrado, creando nuevo...")
                await create_model(model_path)

            self.movie_vectors = joblib.load(model_path)

            vectorizer_path = model_path.replace(".pkl", "_vectorizer.pkl")
            metadata_path = model_path.replace(".pkl", "_metadata.pkl")

            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)

            if os.path.exists(metadata_path):
                self.movie_metadata = joblib.load(metadata_path)
            else:
                self.movie_metadata = pd.DataFrame(
                    {
                        "movie_id": range(len(self.movie_vectors)),
                        "title": [
                            f"Movie_{i}" for i in range(len(self.movie_metadata))
                        ],
                    }
                )

            self._build_title_map()

            print(f"✅ Modelo cargado: {len(self.movie_vectors)} películas")

        except Exception as e:
            print(f"❌ Error inicializando modelo: {e}")
            raise

    def _build_title_map(self):
        """Construye un mapa para búsqueda rápida de títulos."""
        if self.movie_metadata is not None:
            for idx, row in self.movie_metadata.iterrows():
                title_lower = row["title"].lower()
                self.movie_titles_map[title_lower] = idx

    def find_movie_index(self, movie_title: str) -> Optional[int]:
        """
        Encuentra el índice de una película por título.

        Args:
            movie_title: Título de la película

        Returns:
            Índice de la película o None si no se encuentra
        """
        title_lower = movie_title.lower()

        if title_lower in self.movie_titles_map:
            return self.movie_titles_map[title_lower]

        for stored_title, idx in self.movie_titles_map.items():
            if title_lower in stored_title or stored_title in title_lower:
                return idx

        return None

    def get_similar_movies(self, movie_title: str, top_n: int = 5) -> List[dict]:
        """
        Obtiene películas similares a una dada.

        Args:
            movie_title: Título de la película
            top_n: Número de recomendaciones

        Returns:
            Lista de películas similares
        """
        if not self.is_ready():
            raise HTTPException(
                status_code=503, detail="Servicio de recomendación no disponible"
            )

        if self.movie_metadata is None:
            raise HTTPException(
                status_code=503, detail="Metadatos de películas no disponibles"
            )

        movie_idx = self.find_movie_index(movie_title)
        if movie_idx is None:
            raise HTTPException(
                status_code=404, detail=f"Película '{movie_title}' no encontrada"
            )

        similarity_scores = cosine_similarity(
            [self.movie_vectors[movie_idx]], self.movie_vectors
        )[0]

        similar_indices = similarity_scores.argsort()[-(top_n + 1) : -1][::-1]

        results = []
        for idx in similar_indices:
            if self.movie_metadata is not None:
                movie_data = self.movie_metadata.iloc[idx]
                results.append(
                    {
                        "title": movie_data["title"],
                        "movie_id": int(movie_data["movie_id"]),
                        "similarity_score": float(similarity_scores[idx]),
                        "rank": len(results) + 1,
                    }
                )

        return results

    def recommend_by_tags(self, tags: List[str], top_n: int = 5) -> List[dict]:
        """
        Recomienda películas basadas en tags.

        Args:
            tags: Lista de tags
            top_n: Número de recomendaciones

        Returns:
            Lista de películas recomendadas
        """
        if not self.is_ready():
            raise HTTPException(
                status_code=503, detail="Servicio de recomendación no disponible"
            )

        if self.vectorizer is None:
            raise HTTPException(status_code=503, detail="Vectorizador no disponible")

        tags_text = " ".join(tags)
        query_vector = self.vectorizer.transform([tags_text]).toarray()

        similarity_scores = cosine_similarity(query_vector, self.movie_vectors)[0]

        similar_indices = similarity_scores.argsort()[-top_n:][::-1]

        results = []
        for idx in similar_indices:
            movie_data = self.movie_metadata.iloc[idx]
            results.append(
                {
                    "title": movie_data["title"],
                    "movie_id": int(movie_data["movie_id"]),
                    "similarity_score": float(similarity_scores[idx]),
                    "rank": len(results) + 1,
                }
            )

        return results

    def is_ready(self) -> bool:
        """Verifica si el servicio está listo para usar."""
        return all(
            [
                self.movie_vectors is not None,
                self.movie_metadata is not None,
                len(self.movie_vectors) > 0,
            ]
        )

    def get_stats(self) -> dict:
        """Obtiene estadísticas del servicio."""
        return {
            "movies_loaded": len(self.movie_vectors) if self.movie_vectors else 0,
            "vectorizer_loaded": self.vectorizer is not None,
            "title_map_size": len(self.movie_titles_map),
            "service_ready": self.is_ready(),
        }


recommendation_service = RecommendationModelService()
