from sklearn.feature_extraction.text import CountVectorizer
import joblib
import pandas as pd
import numpy as np
from typing import Tuple
from utils.convert import convert, collapse, fetch_director, convert3

class MovieDataPreprocessor:
    """Clase para preprocesar datos de películas de manera estructurada."""
    
    def __init__(self):
        self.required_columns = ['movie_id', 'title', 'overview', 'genres', 
                                 'keywords', 'cast', 'crew']
        self.text_columns = ['overview', 'genres', 'keywords', 'cast', 'crew']

    def load_and_merge_data(self) -> pd.DataFrame:
        """
        Carga y fusiona los datasets de películas y créditos.
        
        Returns:
            DataFrame combinado
        """
        movies_url = "hf://datasets/alejandrowallace/tmdb-5000/tmdb_5000_movies.csv"
        credits_url = "hf://datasets/tshera3/5000_tdmb_movies/tmdb_5000_credits.csv.zip"
        
        movies = pd.read_csv(movies_url)
        credits = pd.read_csv(credits_url)
        
        return movies.merge(credits, on='title', how='inner')
    
    def filter_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra columnas necesarias y elimina valores nulos.
        
        Args:
            df: DataFrame completo
            
        Returns:
            DataFrame limpio y filtrado
        """
        df_clean = df[self.required_columns].copy()
        df_clean.dropna(inplace=True)
        
        return df_clean
    
    def preprocess_text_column(self, df: pd.DataFrame, column: str, 
                               processor_func) -> None:
        """
        Preprocesa una columna de texto con la función especificada.
        
        Args:
            df: DataFrame a procesar
            column: Nombre de la columna
            processor_func: Función de procesamiento
        """
        df[column] = df[column].apply(processor_func)
    
    def process_all_text_columns(self, df: pd.DataFrame) -> None:
        """Procesa todas las columnas de texto según sus funciones específicas."""
        column_processors = {
            'genres': convert,
            'keywords': convert,
            'cast': convert3,
            'crew': fetch_director,
            'overview': lambda x: x.split()
        }
        
        for column, processor in column_processors.items():
            self.preprocess_text_column(df, column, processor)
    
    def collapse_text_columns(self, df: pd.DataFrame) -> None:
        """Aplica la función collapse a las columnas especificadas."""
        columns_to_collapse = ['cast', 'crew', 'genres', 'keywords']
        
        for column in columns_to_collapse:
            df[column] = df[column].apply(collapse)
    
    def create_tags_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea la columna 'tags' combinando todas las características de texto.
        
        Args:
            df: DataFrame con columnas procesadas
            
        Returns:
            DataFrame con columna 'tags' creada
        """
        df = df.copy()
        
        text_features = ['overview', 'genres', 'keywords', 'cast', 'crew']
        df['tags'] = df[text_features].apply(
            lambda row: ' '.join(sum(row.values, [])), axis=1
        )

        columns_to_drop = ['overview', 'cast', 'genres', 'crew', 'keywords']
        df.drop(columns=columns_to_drop, inplace=True)
        
        return df
    
    def preprocess_pipeline(self) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de preprocesamiento.
        
        Returns:
            DataFrame preprocesado listo para vectorización
        """
        df = self.load_and_merge_data()
        df = self.filter_and_clean_data(df)
        self.process_all_text_columns(df)
        self.collapse_text_columns(df)
        df = self.create_tags_feature(df)
        
        return df

class CountVectorizerModel:
    """Clase para manejar la vectorización de texto."""
    
    def __init__(self, max_features: int = 5000, stop_words: str = 'english'):
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words=stop_words
        )
        self.is_fitted = False
    
    def fit_transform(self, text_series: pd.Series) -> np.ndarray:
        """
        Ajusta el vectorizador y transforma los textos.
        
        Args:
            text_series: Serie con textos a vectorizar
            
        Returns:
            Array con vectores transformados
        """
        vectors = self.vectorizer.fit_transform(text_series)
        self.is_fitted = True
        return vectors.toarray()
    
    def save_vectorizer(self, path: str) -> None:
        """Guarda el vectorizador entrenado."""
        if self.is_fitted:
            joblib.dump(self.vectorizer, f"{path}_vectorizer.pkl")
        else:
            raise ValueError("El vectorizador no ha sido entrenado")
    
    def save_vectors(self, vectors: np.ndarray, path: str) -> None:
        """Guarda los vectores transformados."""
        joblib.dump(vectors, path)


def validate_data(df: pd.DataFrame) -> bool:
    """
    Valida que el DataFrame tenga la estructura esperada.
    
    Args:
        df: DataFrame a validar
        
    Returns:
        True si es válido, False en caso contrario
    """
    required_columns = ['movie_id', 'title', 'tags']
    
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        print(f"Columnas faltantes: {missing}")
        return False
    
    if df['tags'].isnull().any():
        print("Advertencia: Hay valores nulos en la columna 'tags'")
        return False
    
    return True

async def create_model(model_path: str, max_features: int = 5000) -> Tuple[np.ndarray, CountVectorizer]:
    """
    Crea un modelo de vectorización a partir de datos de películas.
    
    Args:
        model_path: Ruta donde guardar el modelo
        max_features: Número máximo de features para CountVectorizer
        
    Returns:
        Tupla con (vectores, vectorizador entrenado)
        
    Raises:
        ValueError: Si los datos no son válidos
    """
    try:
        preprocessor = MovieDataPreprocessor()
        movies_processed = preprocessor.preprocess_pipeline()
        
        if not validate_data(movies_processed):
            raise ValueError("Datos no válidos para la vectorización")
        
        vectorizer_model = CountVectorizerModel(max_features=max_features)
        vectors = vectorizer_model.fit_transform(movies_processed['tags'])
        
        joblib.dump(vectors, model_path)
        
        vectorizer_model.save_vectorizer(model_path.replace('.pkl', ''))
        
        metadata = movies_processed[['movie_id', 'title']].copy()
        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"Modelo guardado exitosamente en: {model_path}")
        print(f"Dimensiones de los vectores: {vectors.shape}")
        print(f"Número de películas procesadas: {len(movies_processed)}")
        
        return vectors, vectorizer_model.vectorizer
        
    except Exception as e:
        print(f"Error al crear el modelo: {str(e)}")
        raise