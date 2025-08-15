# Importaciones necesarias
import json
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Manejo automático de CORS

# === CONFIGURACIÓN ===
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

INDEX_FILE = "vector_index.faiss"
METADATA_FILE = "metadata.json"

# === APP FLASK ===
app = Flask(__name__)

# === Habilitar CORS para tu frontend ===
# Esto asegura que cualquier método (POST, OPTIONS) devuelva los headers CORS
CORS(
    app,
    origins="https://www.neuro.uy",
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# === FUNCIONES ===
def obtener_embedding(texto):
    response = genai.embed_content(
        model="models/embedding-001",
        content=texto,
        task_type="RETRIEVAL_QUERY"
    )
    return np.array(response["embedding"], dtype=np.float32)

def cargar_index_y_metadata():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("❌ No se encontraron los archivos FAISS o metadata.json")
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)["metadatos"]
    return index, metadata

def buscar_contexto_para_gemini(consulta, top_k=3):
    index, metadata = cargar_index_y_metadata()
    vector_consulta = obtener_embedding(consulta)
    D, I = index.search(np.array([vector_consulta]), k=top_k)
    contexto = ""
    for idx in I[0]:
        doc = metadata[idx]
        contexto += f"Documento: {doc['documento']}\nTexto: {doc['texto']}\n\n"
    return contexto

def responder_con_gemini(pregunta, contexto):
    modelo = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
Usá el siguiente contexto para responder la pregunta del usuario, si no encuentras la respuesta intenta responderla tu,
ten encuenta que somos una empresa de TI que soluciona problemas a la industria y al agro.

Contexto:
{contexto}

Pregunta:
{pregunta}
"""
    respuesta = modelo.generate_content(prompt)
    return respuesta.text

# === ENDPOINT API ===
@app.route("/consultar", methods=["POST", "OPTIONS"])
def consultar():
    # OPTIONS (preflight) se maneja automáticamente con flask_cors
    if request.method == "OPTIONS":
        return jsonify({}), 200

    # POST
    data = request.get_json()
    pregunta = data.get("pregunta")

    if not pregunta:
        return jsonify({"error": "Falta el campo 'pregunta'"}), 400

    try:
        contexto = buscar_contexto_para_gemini(pregunta)
        respuesta = responder_con_gemini(pregunta, contexto)
        return jsonify({"respuesta": respuesta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === ENDPOINT GLOBAL PARA CUALQUIER PRELIGHT (opcional) ===
@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    # Esto asegura que cualquier ruta con OPTIONS devuelva los headers CORS
    return jsonify({}), 200

# === INICIO LOCAL (OPCIONAL) ===
if __name__ == "__main__":
    app.run(debug=True, port=8000)
