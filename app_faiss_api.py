import json
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
from functools import lru_cache

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from sentence_transformers import SentenceTransformer

# === CONFIGURACIÓN ===
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("⚠️  GEMINI_API_KEY no está definido en las variables de entorno")
genai.configure(api_key=API_KEY)

INDEX_FILE = "vector_index.faiss"
METADATA_FILE = "metadata.json"

# === MODELO DE EMBEDDINGS LOCAL ===
print("⏳ Cargando modelo de embeddings local (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Modelo de embeddings cargado")

# === APP FLASK ===
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

CORS(
    app,
    origins=["https://neuro.uy", "https://www.neuro.uy"],
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# === FUNCIONES ===
def cargar_index_y_metadata():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("❌ No se encontraron los archivos FAISS o metadata.json")
    
    index = faiss.read_index(INDEX_FILE)
    
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        # Aseguramos que es lista de diccionarios
        if not isinstance(metadata, list):
            raise ValueError("❌ metadata.json debe ser una lista de diccionarios")
    
    return index, metadata

# Cargamos FAISS y metadata una sola vez al inicio
try:
    index, metadata = cargar_index_y_metadata()
    print("✅ Índice FAISS y metadata cargados en memoria")
except Exception as e:
    print(f"⚠️ Error cargando FAISS: {e}")
    index, metadata = None, None

@lru_cache(maxsize=5000)
def obtener_embedding_local(texto: str):
    """Devuelve el embedding usando SentenceTransformer con cache"""
    print("📩 Texto a embebear (local):", texto[:200])  # primeros 200 chars
    vector = embedding_model.encode([texto], convert_to_numpy=True)[0]
    return np.array(vector, dtype=np.float32)

def buscar_contexto_para_gemini(consulta, top_k=3):
    if index is None or metadata is None:
        raise RuntimeError("⚠️ El índice FAISS no está disponible en memoria")

    vector_consulta = obtener_embedding_local(consulta)
    D, I = index.search(np.array([vector_consulta]), k=top_k)

    contexto = ""
    for idx in I[0]:
        doc = metadata[idx]
        contexto += f"Documento: {doc['file']}\nTexto: {doc['text']}\n\n"
    return contexto

def responder_con_gemini(pregunta, contexto):
    modelo = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
Usá el siguiente contexto para responder la pregunta del usuario, si no encuentras la respuesta intenta responderla tú,
ten en cuenta que somos una empresa de TI que soluciona problemas a la industria y al agro.

Contexto:
{contexto}

Pregunta:
{pregunta}
"""
    respuesta = modelo.generate_content(prompt)

    # Manejo robusto de salida
    if hasattr(respuesta, "text"):
        return respuesta.text
    elif hasattr(respuesta, "candidates"):
        return respuesta.candidates[0].content.parts[0].text
    else:
        raise ValueError(f"❌ No se pudo interpretar la respuesta de Gemini: {respuesta}")

# === ENDPOINT PRINCIPAL ===
@app.route("/consultar", methods=["POST", "OPTIONS"])
def consultar():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json()
    pregunta = data.get("pregunta")
    if not pregunta:
        return jsonify({"error": "Falta el campo 'pregunta'"}), 400

    try:
        contexto = buscar_contexto_para_gemini(pregunta)
        respuesta = responder_con_gemini(pregunta, contexto)
        return jsonify({"respuesta": respuesta})
    except Exception as e:
        print("❌ ERROR en /consultar:", e)  # Log visible en Railway
        return jsonify({"error": str(e)}), 500

@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    return jsonify({}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
