# Importaciones necesarias para la aplicación Flask
import json
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

from flask import Flask, request, jsonify

# No necesitas importar 'CORS' ni 'make_response' si manejas los encabezados manualmente
# from flask_cors import CORS
# from flask import make_response

# === CONFIGURACIÓN ===
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

INDEX_FILE = "vector_index.faiss"
METADATA_FILE = "metadata.json"

# === APP FLASK ===
app = Flask(__name__)

# === MANEJO DE CORS GLOBAL Y ROBUSTO ===
# Este decorador se ejecuta DESPUÉS de cada petición y agrega los encabezados
# CORS a CADA respuesta, independientemente del método (POST, OPTIONS, etc.).
# Esto asegura que el encabezado esté siempre presente.
@app.after_request
def add_cors_headers(response):
    # Agrega el encabezado que permite el acceso desde el origen de tu frontend
    response.headers["Access-Control-Allow-Origin"] = "https://www.neuro.uy"
    # Agrega los métodos HTTP que están permitidos
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    # Agrega los encabezados que el cliente puede enviar
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# === FUNCIONES ===
def obtener_embedding(texto):
    """
    Genera el embedding de un texto usando el modelo de Google.
    """
    response = genai.embed_content(
        model="models/embedding-001",
        content=texto,
        task_type="RETRIEVAL_QUERY"
    )
    return np.array(response["embedding"], dtype=np.float32)

def cargar_index_y_metadata():
    """
    Carga el índice FAISS y los metadatos desde archivos.
    """
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("❌ No se encontraron los archivos FAISS o metadata.json")

    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)["metadatos"]
    return index, metadata

def buscar_contexto_para_gemini(consulta, top_k=3):
    """
    Busca los documentos más relevantes en el índice FAISS para una consulta.
    """
    index, metadata = cargar_index_y_metadata()
    vector_consulta = obtener_embedding(consulta)
    D, I = index.search(np.array([vector_consulta]), k=top_k)

    contexto = ""
    for idx in I[0]:
        doc = metadata[idx]
        contexto += f"Documento: {doc['documento']}\nTexto: {doc['texto']}\n\n"
    return contexto

def responder_con_gemini(pregunta, contexto):
    """
    Usa el modelo Gemini para generar una respuesta basada en el contexto y la pregunta.
    """
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
# Ahora el endpoint solo necesita manejar la petición POST, el CORS se gestiona globalmente.
@app.route("/consultar", methods=["POST"])
def consultar():
    """
    Endpoint principal para recibir una pregunta y devolver una respuesta de Gemini.
    """
    data = request.get_json()
    pregunta = data.get("pregunta")

    if not pregunta:
        response = jsonify({"error": "Falta el campo 'pregunta'"})
        return response, 400

    try:
        contexto = buscar_contexto_para_gemini(pregunta)
        respuesta = responder_con_gemini(pregunta, contexto)
        response = jsonify({"respuesta": respuesta})
        return response
    except Exception as e:
        response = jsonify({"error": str(e)})
        return response, 500

# === INICIO LOCAL (OPCIONAL) ===
if __name__ == "__main__":
    app.run(debug=True, port=8000)
