from flask import Flask, request
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

# 🔹 Cargar variables desde .env
load_dotenv()

app = Flask(__name__)
CORS(app)  # habilita CORS para todos los orígenes

# 🔹 Configuración desde variables de entorno
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        # Verificación inicial de Meta
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if token == VERIFY_TOKEN:
            return challenge
        return "Token inválido", 403

    if request.method == "POST":
        data = request.get_json()

        # 📩 Procesar mensajes entrantes
        if data.get("entry"):
            for entry in data["entry"]:
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    messages = value.get("messages", [])
                    if messages:
                        msg = messages[0]
                        from_number = msg.get("from")  # número del remitente
                        text = msg.get("text", {}).get("body", "")

                        print(f"📩 Mensaje de {from_number}: {text}")

                        # Responder automáticamente
                        send_message(from_number, f"Recibí tu mensaje: '{text}' ✅")

        return "EVENT_RECEIVED", 200

def send_message(to, message):
    """Envia un mensaje de texto a través de la Cloud API"""
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    r = requests.post(url, json=payload, headers=headers)
    print(f"📤 Enviado a {to}: {r.status_code} - {r.text}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
