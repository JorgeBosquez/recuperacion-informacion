import os

import chromadb
import gradio as gr
from google import genai
from sentence_transformers import CrossEncoder, SentenceTransformer

MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
MODELO_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODELO_GEMINI = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
RUTA_CHROMA = "chroma_db"
NOMBRE_COLECCION = "arxiv_papers"

RESPUESTA_FUERA_DE_DOMINIO = (
    "Lo siento, esta consulta no está relacionada con el dominio "
    "del corpus de artículos científicos de arXiv o no existe "
    "información suficiente para responderla."
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("No se encontró la variable GEMINI_API_KEY.")

cliente_gemini = genai.Client(api_key=api_key)
modelo_embeddings = SentenceTransformer(MODELO_EMBEDDINGS)
modelo_reranker = CrossEncoder(MODELO_RERANKER)
cliente_chroma = chromadb.PersistentClient(path=RUTA_CHROMA)
coleccion = cliente_chroma.get_collection(name=NOMBRE_COLECCION)


def buscar_documentos(consulta, top_k=15):
    consulta = consulta.strip()
    embedding_consulta = modelo_embeddings.encode(
        consulta,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    resultados = coleccion.query(
        query_embeddings=[embedding_consulta.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documentos = []
    for posicion, paper_id in enumerate(resultados["ids"][0]):
        metadata = resultados["metadatas"][0][posicion] or {}
        distancia = float(resultados["distances"][0][posicion])
        documentos.append({
            "paper_id": paper_id,
            "title": metadata.get("title", ""),
            "categories": metadata.get("categories", ""),
            "abstract": metadata.get("abstract", ""),
            "document": resultados["documents"][0][posicion],
            "similarity": 1.0 - distancia,
        })
    return documentos


def rerank_documentos(consulta, documentos, top_n=5):
    if not documentos:
        return []
    pares = [[consulta, d["document"]] for d in documentos]
    scores = modelo_reranker.predict(pares, show_progress_bar=False)
    salida = []
    for documento, score in zip(documentos, scores):
        copia = documento.copy()
        copia["rerank_score"] = float(score)
        salida.append(copia)
    salida.sort(key=lambda x: x["rerank_score"], reverse=True)
    return salida[:top_n]


def construir_contexto(documentos, max_caracteres=5000):
    bloques = []
    total = 0
    for i, d in enumerate(documentos, start=1):
        bloque = (
            f"[Document {i}]\n"
            f"Title: {d.get('title', '')}\n"
            f"Categories: {d.get('categories', '')}\n"
            f"Abstract: {d.get('abstract', '')}\n"
        )
        if total + len(bloque) > max_caracteres:
            break
        bloques.append(bloque)
        total += len(bloque)
    return "\n".join(bloques)


def generar_respuesta(prompt):
    respuesta = cliente_gemini.models.generate_content(
        model=MODELO_GEMINI,
        contents=prompt,
    )
    texto = getattr(respuesta, "text", None)
    return texto.strip() if texto else "The corpus does not contain enough information to answer this question."


def responder_rag(consulta, top_k=15, top_n=5, umbral_similitud=0.30, umbral_reranking=-2.0):
    if not isinstance(consulta, str) or not consulta.strip():
        raise ValueError("La consulta no puede estar vacía.")

    candidatos = buscar_documentos(consulta, top_k=top_k)
    documentos = rerank_documentos(consulta, candidatos, top_n=top_n)

    if not documentos:
        return {"respuesta": RESPUESTA_FUERA_DE_DOMINIO, "evidencias": [], "fuera_de_dominio": True}

    mejor = documentos[0]
    fuera = mejor.get("similarity", 0) < umbral_similitud or mejor.get("rerank_score", float("-inf")) < umbral_reranking
    if fuera:
        return {"respuesta": RESPUESTA_FUERA_DE_DOMINIO, "evidencias": documentos, "fuera_de_dominio": True}

    contexto = construir_contexto(documentos)
    prompt = f"""
You are an academic assistant specialized in scientific papers.
Answer the question using ONLY the retrieved context.
- Answer directly and clearly in 3 to 5 complete sentences.
- Do not invent information or use external knowledge.
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
    }


def formatear_evidencias(evidencias):
    if not evidencias:
        return "No se encontraron evidencias."
    bloques = []
    for i, e in enumerate(evidencias, start=1):
        bloques.append(f"""
### Documento {i}

**Título:** {e.get('title', 'No disponible')}

**Paper ID:** `{e.get('paper_id', 'No disponible')}`

**Categorías:** {e.get('categories', 'No disponible')}

**Similitud:** `{e.get('similarity', 0.0):.4f}`

**Score de re-ranking:** `{e.get('rerank_score', 0.0):.4f}`

**Abstract:**

{e.get('abstract', '')[:800]}
""")
    return "\n\n---\n\n".join(bloques)


def chat(consulta):
    try:
        resultado = responder_rag(consulta)
        if resultado.get("fuera_de_dominio", False):
            evidencias = "⚠️ La consulta fue identificada como fuera del dominio del corpus científico."
        else:
            evidencias = formatear_evidencias(resultado.get("evidencias", []))
        return resultado["respuesta"], evidencias
    except Exception as error:
        return f"No fue posible procesar la consulta: {error}", "No se pudieron mostrar evidencias."


demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(
        label="Consulta",
        lines=3,
        placeholder="Example: What are the main applications of Graph Neural Networks?",
    ),
    outputs=[
        gr.Textbox(label="Respuesta generada", lines=8),
        gr.Markdown(),
    ],
    title="Sistema RAG sobre artículos científicos de arXiv",
    description="Recuperación semántica, re-ranking, detección fuera del dominio y generación fundamentada mediante Gemini.",
    examples=[
        ["What are the main applications of Graph Neural Networks?"],
        ["How is reinforcement learning used in robotics?"],
        ["Recent advances in diffusion models for image generation."],
    ],
)

if __name__ == "__main__":
    demo.launch()
