# Importaciones necesarias para la aplicación Flask
import json
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

from flask import Flask, request, jsonify
from flask_cors import CORS # Importación de la extensión Flask-CORS

# === CONFIGURACIÓN ===
# Carga las variables de entorno del archivo .env
load_dotenv()
# Obtiene la clave de la API de Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
# Configura la API de Gemini
genai.configure(api_key=API_KEY)

# Nombres de los archivos para el índice FAISS y los metadatos
INDEX_FILE = "vector_index.faiss"
METADATA_FILE = "metadata.json"

# === APP FLASK ===
app = Flask(__name__)
# Configuración de CORS con la extensión.
# Esto es la forma más limpia de resolver el problema.
# Permite que el dominio "https://www.neuro.uy" acceda al endpoint "/consultar".
# Flask-CORS se encargará automáticamente de las peticiones OPTIONS (preflight).
CORS(app, resources={r"/consultar": {"origins": "https://www.neuro.uy"}})

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
        # Carga los metadatos del archivo JSON
        metadata = json.load(f)["metadatos"]
    return index, metadata

def buscar_contexto_para_gemini(consulta, top_k=3):
    """
    Busca los documentos más relevantes en el índice FAISS para una consulta.
    """
    index, metadata = cargar_index_y_metadata()
    vector_consulta = obtener_embedding(consulta)
    # Realiza la búsqueda en el índice
    D, I = index.search(np.array([vector_consulta]), k=top_k)

    contexto = ""
    # Construye el contexto a partir de los documentos encontrados
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
    # Genera la respuesta
    respuesta = modelo.generate_content(prompt)
    return respuesta.text

# === ENDPOINT API ===
# El endpoint ahora solo necesita manejar la petición POST, ya que la extensión
# Flask-CORS se encarga de la preflight OPTIONS.
@app.route("/consultar", methods=["POST"])
def consultar():
    """
    Endpoint principal para recibir una pregunta y devolver una respuesta de Gemini.
    """
    data = request.get_json()
    pregunta = data.get("pregunta")

    if not pregunta:
        # Devuelve un error si no se proporciona una pregunta
        response = jsonify({"error": "Falta el campo 'pregunta'"})
        return response, 400

    try:
        # Busca el contexto, genera la respuesta y la devuelve
        contexto = buscar_contexto_para_gemini(pregunta)
        respuesta = responder_con_gemini(pregunta, contexto)
        response = jsonify({"respuesta": respuesta})
        return response
    except Exception as e:
        # Manejo de errores y devolución de un mensaje de error
        response = jsonify({"error": str(e)})
        return response, 500

# === INICIO LOCAL (OPCIONAL) ===
if __name__ == "__main__":
    # Inicia el servidor Flask en modo de depuración
    app.run(debug=True, port=8000)