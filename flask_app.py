from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from queue import Queue
from threading import Thread
from PIL import Image
from gradio_client import Client, file

app = Flask(__name__)

# Ensure directories exist
if not os.path.exists("./compressed_image"):
    os.makedirs("./compressed_image")

frame_queue = Queue()
client = None  # Initialize the client as None to configure later based on user input
user_prompt = ""

def generate_image_name():
    return f'./compressed_image/temp.jpg'

def compress_image(frame, quality=50):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    convert_pil = Image.fromarray(frame_rgb)
    output_image_path = generate_image_name()
    convert_pil.save(output_image_path, optimize=True, quality=quality)
    return output_image_path

def describe_frame(frame):
    global client, user_prompt
    if not client or not user_prompt:
        return "Client not initialized or prompt missing"
    
    compressed_image_path = compress_image(frame)
    try:
        result = client.predict(user_prompt, file(compressed_image_path), api_name="/predict")
        return result
    except Exception as e:
        print("Error during prediction:", e)
        return "Error during prediction"

def process_frames():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        result = describe_frame(frame)
        frame_queue.task_done()

Thread(target=process_frames, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_params', methods=['POST'])
def set_params():
    global client, user_prompt
    data = request.json
    app_url = data.get('app_url')
    user_prompt = data.get('prompt')
    
    try:
        # Initialize client with user-provided app_url
        client = Client(app_url, auth=["admin", "admin"])
        return jsonify({"status": "success"})
    except Exception as e:
        print("Error setting parameters:", e)
        return jsonify({"status": "error", "message": str(e)})

@app.route('/upload', methods=['POST'])
def upload_frame():
    data = request.files['frame']
    frame = cv2.imdecode(np.frombuffer(data.read(), np.uint8), cv2.IMREAD_COLOR)
    
    result = describe_frame(frame)  # Get description immediately after processing
    
    return jsonify({"status": "success", "result": result})  # Return result along with status

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
