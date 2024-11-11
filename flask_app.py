import os
import cv2
import threading
from queue import Queue
from PIL import Image
from gradio_client import Client, file
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, Response, jsonify

# Flask app
app = Flask(__name__)

# Gradio server URL (Ensure this URL is active)
app_url = "https://3acfc3ffd46eb820b5.gradio.live/"  # Replace with active URL from Colab
username = "admin"
password = "admin"

# Connect to the Gradio server
try:
    client = Client(app_url, auth=[str(username), str(password)])
    print("Connected to the Gradio server successfully.")
except Exception as e:
    print("Error: Could not connect to the server. Please make sure the URL is correct and the server is running.")
    print(e)
    exit(1)

# Ensure directories exist
if not os.path.exists("./compressed_image"):
    os.makedirs("./compressed_image")

# Queue to handle frames for description
frame_queue = Queue()
output_result = None  # Store the Gradio inference result here

# Generate a unique image file name
def generate_image_name():
    return f'./compressed_image/temp.jpg'

# Compress the image to reduce file size
def compress_image(frame, quality=50):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    convert_pil = Image.fromarray(frame_rgb)
    output_image_path = generate_image_name()
    convert_pil.save(output_image_path, optimize=True, quality=quality)
    return output_image_path

# Function to describe the frame image
def describe_frame(frame):
    global output_result
    compressed_image_path = compress_image(frame)
    prompt = "Describe the Image "

    try:
        result = client.predict(prompt, file(compressed_image_path), api_name="/predict")
        print("Result:", result)
        output_result = result  # Update the result for display
    except Exception as e:
        print("Error during prediction:", e)

# Background function to process frames with limited concurrency
def process_frames():
    with ThreadPoolExecutor(max_workers=3) as executor:  # Limit to 3 concurrent threads
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            executor.submit(describe_frame, frame)
            frame_queue.task_done()

# Start the background thread for frame processing
threading.Thread(target=process_frames, daemon=True).start()

# Capture video feed
def generate_video():
    cap = cv2.VideoCapture(0)
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Increment frame counter and process every 10th frame
        frame_counter += 1
        if frame_counter == 60:
            frame_queue.put(frame)
            frame_counter = 0

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame in byte format for HTML video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route to display video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get the latest result
@app.route('/get_result')
def get_result():
    global output_result
    return jsonify({"result": output_result if output_result else "Processing..."})

# Homepage route
@app.route('/')
def index():
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
