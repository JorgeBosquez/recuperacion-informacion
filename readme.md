# Recuperación de Información: TF-IDF y BM25

## Descripción

Este proyecto corresponde a la resolución de los ejercicios:

- Ejercicio 03: Modelo Vectorial TF-IDF
- Ejercicio 04: Modelo Probabilístico BM25

El objetivo es implementar y comparar dos enfoques fundamentales en recuperación de información utilizando un corpus de documentos reales.

------------------------------------------------------------

MODELOS IMPLEMENTADOS

TF-IDF (Modelo Vectorial)

TF-IDF (Term Frequency - Inverse Document Frequency) permite representar documentos como vectores numéricos, donde cada término tiene un peso basado en:

- TF: cuántas veces aparece una palabra en un documento
- IDF: qué tan común es esa palabra en el corpus

Se utiliza la similitud coseno para medir la relevancia entre consulta y documentos.

------------------------------------------------------------

BM25 (Modelo Probabilístico)

BM25 es una mejora de TF-IDF que introduce:

- Saturación de términos
- Normalización por longitud del documento
- Modelo probabilístico de relevancia

Esto permite obtener resultados más precisos.

------------------------------------------------------------

ESTRUCTURA DEL PROYECTO

03tfidf.ipynb
04bm25.ipynb
README.md

------------------------------------------------------------

DATASET

El dataset utilizado corresponde a libros de Project Gutenberg.

Acceso al dataset:
https://drive.google.com/drive/folders/133fRRkPFZzhFhSp52m2ijQ5GOO3S-tzx

------------------------------------------------------------

REQUISITOS

Instalar librerías con:

pip install pandas scikit-learn matplotlib

------------------------------------------------------------

EJECUCIÓN

1. Descargar el dataset
2. Colocar la carpeta en el mismo directorio del proyecto
3. Abrir los notebooks en VS Code o Jupyter
4. Ejecutar en orden

------------------------------------------------------------

RESULTADOS

- TF-IDF recupera documentos relevantes según términos importantes
- BM25 mejora los resultados considerando longitud y saturación
- BM25 produce rankings más equilibrados

------------------------------------------------------------

CONCLUSIÓN

BM25 es más robusto que TF-IDF para recuperación de información en colecciones grandes.

TF-IDF es un buen modelo base, pero BM25 mejora la precisión de los resultados.

------------------------------------------------------------

AUTOR

Jorge Bósquez
