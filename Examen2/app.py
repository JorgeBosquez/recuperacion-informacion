import os
import re

import chromadb
import gradio as gr
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURACIÓN
# ============================================================

load_dotenv()

MODELO_GEMINI = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
RUTA_CHROMA = "chroma_db_demo"
NOMBRE_COLECCION = "arxiv_papers"

TOP_K_RECUPERACION = 15
TOP_N_RERANKING = 5
UMBRAL_SIMILITUD = 0.30
UMBRAL_RERANKING = 0.20

RESPUESTA_FUERA_DE_DOMINIO = (
    "Lo siento, esta consulta no está relacionada con el dominio "
    "del corpus de artículos científicos de arXiv o no existe "
    "información suficiente para responderla."
)


# ============================================================
# GEMINI
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "No se encontró GEMINI_API_KEY. "
        "Configúrala en .env o en las variables de entorno."
    )

cliente_gemini = genai.Client(api_key=api_key)


# ============================================================
# CHROMADB
# ============================================================

embedding_function = embedding_functions.DefaultEmbeddingFunction()

cliente_chroma = chromadb.PersistentClient(path=RUTA_CHROMA)

coleccion = cliente_chroma.get_collection(
    name=NOMBRE_COLECCION,
    embedding_function=embedding_function,
)


# ============================================================
# RECUPERACIÓN SEMÁNTICA
# ============================================================

def buscar_documentos(consulta: str, top_k: int = TOP_K_RECUPERACION):
    if not isinstance(consulta, str):
        raise TypeError("La consulta debe ser texto.")

    consulta = consulta.strip()

    if not consulta:
        raise ValueError("La consulta no puede estar vacía.")

    resultados = coleccion.query(
        query_texts=[consulta],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documentos = []

    ids = resultados.get("ids", [[]])[0]
    documents = resultados.get("documents", [[]])[0]
    metadatas = resultados.get("metadatas", [[]])[0]
    distances = resultados.get("distances", [[]])[0]

    for posicion, paper_id in enumerate(ids):
        metadata = metadatas[posicion] or {}
        distancia = float(distances[posicion])

        documentos.append(
            {
                "paper_id": paper_id,
                "title": metadata.get("title", ""),
                "categories": metadata.get("categories", ""),
                "abstract": metadata.get("abstract", ""),
                "document": documents[posicion],
                "distance": distancia,
                "similarity": 1.0 - distancia,
            }
        )

    return documentos


# ============================================================
# RE-RANKING LIGERO
# ============================================================

def normalizar_palabras(texto: str):
    return set(re.findall(r"[a-zA-Z0-9]+", str(texto).lower()))


def rerank_documentos(
    consulta: str,
    documentos: list,
    top_n: int = TOP_N_RERANKING,
):
    if not documentos:
        return []

    palabras_consulta = normalizar_palabras(consulta)
    salida = []

    for documento in documentos:
        texto_documento = (
            documento.get("title", "")
            + " "
            + documento.get("abstract", "")
        )

        palabras_documento = normalizar_palabras(texto_documento)

        if palabras_consulta:
            coincidencia_lexica = (
                len(palabras_consulta & palabras_documento)
                / len(palabras_consulta)
            )
        else:
            coincidencia_lexica = 0.0

        copia = documento.copy()
        copia["rerank_score"] = (
            0.85 * copia.get("similarity", 0.0)
            + 0.15 * coincidencia_lexica
        )
        salida.append(copia)

    salida.sort(
        key=lambda documento: documento["rerank_score"],
        reverse=True,
    )

    return salida[:top_n]


# ============================================================
# CONTEXTO Y GENERACIÓN
# ============================================================

def construir_contexto(documentos: list, max_caracteres: int = 5000):
    bloques = []
    total = 0

    for indice, documento in enumerate(documentos, start=1):
        bloque = (
            f"[Document {indice}]\n"
            f"Title: {documento.get('title', '')}\n"
            f"Categories: {documento.get('categories', '')}\n"
            f"Abstract: {documento.get('abstract', '')}\n"
        )

        if total + len(bloque) > max_caracteres:
            break

        bloques.append(bloque)
        total += len(bloque)

    return "\n".join(bloques)


def generar_respuesta(prompt: str):
    respuesta = cliente_gemini.models.generate_content(
        model=MODELO_GEMINI,
        contents=prompt,
    )

    texto = getattr(respuesta, "text", None)

    if not texto or not texto.strip():
        return (
            "The corpus does not contain enough information "
            "to answer this question."
        )

    return texto.strip()


# ============================================================
# PIPELINE RAG
# ============================================================

def responder_rag(
    consulta: str,
    top_k: int = TOP_K_RECUPERACION,
    top_n: int = TOP_N_RERANKING,
    umbral_similitud: float = UMBRAL_SIMILITUD,
    umbral_reranking: float = UMBRAL_RERANKING,
):
    if not isinstance(consulta, str) or not consulta.strip():
        raise ValueError("La consulta no puede estar vacía.")

    consulta = consulta.strip()

    candidatos = buscar_documentos(consulta, top_k=top_k)
    documentos = rerank_documentos(consulta, candidatos, top_n=top_n)

    if not documentos:
        return {
            "respuesta": RESPUESTA_FUERA_DE_DOMINIO,
            "evidencias": [],
            "fuera_de_dominio": True,
        }

    mejor = documentos[0]
    mejor_similitud = float(mejor.get("similarity", 0.0))
    mejor_rerank = float(mejor.get("rerank_score", 0.0))

    fuera_de_dominio = (
        mejor_similitud < umbral_similitud
        or mejor_rerank < umbral_reranking
    )

    if fuera_de_dominio:
        return {
            "respuesta": RESPUESTA_FUERA_DE_DOMINIO,
            "evidencias": documentos,
            "fuera_de_dominio": True,
            "mejor_similitud": mejor_similitud,
            "mejor_rerank_score": mejor_rerank,
        }

    contexto = construir_contexto(documentos)

    if not contexto.strip():
        return {
            "respuesta": RESPUESTA_FUERA_DE_DOMINIO,
            "evidencias": documentos,
            "fuera_de_dominio": True,
        }

    prompt = f"""
You are an academic assistant specialized in scientific papers.

Answer the question using ONLY the retrieved context.

Instructions:
- Answer directly and clearly in 3 to 5 complete sentences.
- Focus only on what the question asks.
- Do not invent information.
- Do not use external knowledge.
- Cite evidence using [Document 1], [Document 2], etc.
- If the context is insufficient, answer exactly:
  "The corpus does not contain enough information to answer this question."

Context:
{contexto}

Question:
{consulta}

Final answer:
"""

    return {
        "respuesta": generar_respuesta(prompt),
        "evidencias": documentos,
        "fuera_de_dominio": False,
        "mejor_similitud": mejor_similitud,
        "mejor_rerank_score": mejor_rerank,
    }


# ============================================================
# INTERFAZ
# ============================================================

def formatear_evidencias(evidencias: list):
    if not evidencias:
        return "No se encontraron evidencias."

    bloques = []

    for indice, evidencia in enumerate(evidencias, start=1):
        bloques.append(
            f"""
### Documento {indice}

**Título:** {evidencia.get('title', 'No disponible')}

**Paper ID:** `{evidencia.get('paper_id', 'No disponible')}`

**Categorías:** {evidencia.get('categories', 'No disponible')}

**Similitud:** `{evidencia.get('similarity', 0.0):.4f}`

**Score de re-ranking:** `{evidencia.get('rerank_score', 0.0):.4f}`

**Abstract:**

{evidencia.get('abstract', '')[:800]}
"""
        )

    return "\n\n---\n\n".join(bloques)


def chat(consulta: str):
    try:
        resultado = responder_rag(consulta)

        if resultado.get("fuera_de_dominio", False):
            evidencias = (
                "⚠️ La consulta fue identificada como fuera "
                "del dominio del corpus científico."
            )
        else:
            evidencias = formatear_evidencias(
                resultado.get("evidencias", [])
            )

        return resultado["respuesta"], evidencias

    except Exception as error:
        mensaje = str(error)

        if "503" in mensaje or "UNAVAILABLE" in mensaje:
            return (
                "Gemini está temporalmente saturado. "
                "Vuelva a intentarlo en unos segundos.",
                "La recuperación se realizó, pero Gemini no estuvo disponible.",
            )

        return (
            f"No fue posible procesar la consulta: {error}",
            "No se pudieron mostrar evidencias.",
        )


demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(
        label="Consulta",
        lines=3,
        placeholder=(
            "Example: What are the main applications "
            "of Graph Neural Networks?"
        ),
    ),
    outputs=[
        gr.Textbox(label="Respuesta generada", lines=8),
        gr.Markdown(),
    ],
    title="Sistema RAG sobre artículos científicos de arXiv",
    description=(
        "Recuperación semántica, re-ranking ligero, detección "
        "fuera del dominio y generación mediante Gemini."
    ),
    examples=[
        ["What are the main applications of Graph Neural Networks?"],
        ["How is reinforcement learning used in robotics?"],
        ["Recent advances in diffusion models for image generation."],
    ],
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )