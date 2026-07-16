# Sistema RAG sobre artículos científicos de arXiv

Proyecto de Recuperación de Información que implementa un sistema RAG completo sobre un corpus de resúmenes científicos de arXiv.

## Funcionalidades

- Preparación y limpieza del corpus.
- Embeddings con `all-MiniLM-L6-v2`.
- Base vectorial persistente con ChromaDB.
- Recuperación semántica.
- Re-ranking con CrossEncoder.
- Detección de consultas fuera del dominio.
- Generación de respuestas con Gemini.
- Presentación de evidencias.
- Interfaz web con Gradio.

## Ejecución local

```bash
pip install -r requirements.txt
python app.py
```

La variable `GEMINI_API_KEY` debe configurarse como variable de entorno. En Hugging Face Spaces debe agregarse como Secret.

## Entrega

- Repositorio: agregar aquí el enlace de GitHub.
- Aplicación web: agregar aquí el enlace de Hugging Face Spaces.
