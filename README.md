# 🎬 Movie Recommendation API

Una API RESTful construida con FastAPI para recomendar películas basadas en similitud de contenido utilizando procesamiento de lenguaje natural y aprendizaje automático.

## 📋 Descripción

Esta API permite a los usuarios obtener recomendaciones de películas basadas en:
- **Similitud con una película específica** (basado en contenido)
- **Tags o palabras clave** proporcionadas por el usuario
- **Búsqueda** en el catálogo de películas
- **Comparación** de similitud entre dos películas

El sistema utiliza un modelo de similitud coseno sobre vectores TF-IDF generados a partir de metadatos de películas (géneros, actores, director, palabras clave y sinopsis).

## 🚀 Características

- **API RESTful** completa con documentación automática (Swagger/ReDoc)
- **Procesamiento de lenguaje natural** con scikit-learn
- **Sistema de recomendación basado en contenido**
- **Búsqueda y comparación de películas**
- **Singleton pattern** para gestión eficiente del modelo
- **Validación de datos** con Pydantic
- **Arquitectura modular** y escalable
- **Endpoints para monitoreo** y salud del servicio

## 🏗️ Arquitectura

```
src/
├── api/v1/endpoints/         # Endpoints de la API
│   ├── health.py            # Health checks
│   ├── recommend.py         # Recomendaciones por título/tags
│   ├── search.py            # Búsqueda de películas
│   └── compare.py           # Comparación de películas
├── schemas/                 # Esquemas Pydantic
├── services/                # Lógica de negocio
│   ├── model_service.py     # Servicio Singleton del modelo
│   └── model_creator_service.py # Creador del modelo
├── utils/                   # Utilidades
│   └── convert.py          # Funciones de preprocesamiento
├── models/                  # Modelos serializados
└── main.py                 # Aplicación principal FastAPI
```

## 🛠️ Tecnologías

- **Python 3.8+**
- **FastAPI** - Framework web moderno y rápido
- **scikit-learn** - Machine learning y NLP
- **pandas** - Manipulación de datos
- **joblib** - Serialización de modelos
- **NumPy** - Cálculos numéricos
- **Pydantic** - Validación de datos
- **Uvicorn** - Servidor ASGI

## 📦 Instalación

### Prerrequisitos

```bash
python >= 3.8
pip >= 20.0
```

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd movies-recomendation-api
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Crear un archivo `.env` en la raíz del proyecto:

```env
MODEL_PATH=./models/recomendation_model.pkl
MAX_FEATURES=5000
DEBUG=False
```

## 🚀 Uso

### Iniciar la API

```bash
# Modo desarrollo (con recarga automática)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Modo producción
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Acceder a la documentación

Una vez iniciada la API, accede a:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Root**: http://localhost:8000/

## 📡 Endpoints

### Health Check
```http
GET /api/v1/health
```
Verifica el estado del servicio y si el modelo está cargado.

### Recomendaciones por Título
```http
GET /api/v1/recommend/by-title?title=Inception&limit=5
```
Obtiene películas similares a la especificada.

### Recomendaciones por Tags
```http
POST /api/v1/recommend/by-tags
Content-Type: application/json

{
  "tags": ["sci-fi", "space", "adventure"],
  "limit": 5
}
```

### Búsqueda de Películas
```http
GET /api/v1/search?query=avengers&limit=10
```

### Comparación de Películas
```http
GET /api/v1/compare?movie1=Inception&movie2=Matrix
```

## 🔧 Ejemplos de Uso

### Usando cURL

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Recomendaciones para "Inception"
curl "http://localhost:8000/api/v1/recommend/by-title?title=Inception&limit=5"

# Búsqueda
curl "http://localhost:8000/api/v1/search?query=star&limit=3"

# Comparación
curl "http://localhost:8000/api/v1/compare?movie1=Avatar&movie2=Titanic"
```

### Usando Python

```python
import requests

# Configurar base URL
BASE_URL = "http://localhost:8000/api/v1"

# Obtener recomendaciones
response = requests.get(f"{BASE_URL}/recommend/by-title", params={
    "title": "The Dark Knight",
    "limit": 3
})

if response.status_code == 200:
    data = response.json()
    print(f"Recomendaciones para {data['query']}:")
    for rec in data['recommendations']:
        print(f"  - {rec['title']} (similitud: {rec['similarity_score']:.2f})")
```

## 🔍 Cómo Funciona

### Pipeline de Procesamiento

1. **Extracción de Datos**: 
   - Carga datasets de TMDB (5000 películas)
   - Combina información de películas y créditos

2. **Preprocesamiento**:
   - Convierte JSON a listas de strings
   - Extrae géneros, actores, director, palabras clave
   - Tokeniza sinopsis

3. **Vectorización**:
   - Crea una columna "tags" combinando todas las características
   - Aplica CountVectorizer con 5000 features máximos
   - Elimina stop words en inglés

4. **Cálculo de Similitud**:
   - Utiliza similitud coseno entre vectores
   - Encuentra los vecinos más cercanos para cada película

### Algoritmo de Recomendación

```
Película de entrada → Vectorización → Cálculo de similitud coseno 
→ Ordenar por similitud → Filtrar top N → Devolver resultados
```

## 📊 Dataset

El proyecto utiliza el dataset [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata) que contiene:

- **5000 películas** con información detallada
- **Géneros**, palabras clave, sinopsis
- **Reparto** y equipo de producción
- **Puntajes** y metadatos

## 🔄 Actualización del Modelo

Para recrear el modelo con nuevos datos:

```python
from services.model_creator_service import create_model

# Crear y guardar nuevo modelo
await create_model('models/recomendation_model.pkl')
```

## ⚠️ Limitaciones

- Solo recomienda entre las 5000 películas del dataset
- Basado únicamente en similitud de contenido (no colaborativo)
- No considera ratings de usuarios
- Los resultados dependen de la calidad de los metadatos

## 🤝 Contribuir

1. Fork el proyecto
2. Crear una rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

---

⭐ Si te gusta este proyecto, ¡dale una estrella en GitHub!
