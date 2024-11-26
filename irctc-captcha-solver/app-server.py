#$ export KMP_DUPLICATE_LIB_OK=TRUE

import argparse
import base64
import io
import os
from datetime import datetime, time  # Import time to define ranges

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO

# Set the KMP_DUPLICATE_LIB_OK environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# DataSet Directory
TRAIN_DIR = "datasets/train/images"
TEST_DIR = "datasets/test/images"
TRAIN_LABELS = "datasets/train/labels.csv"
TEST_LABELS = "datasets/test/labels.csv"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
# Function to load the YOLO model
def load_model(model_path):
  """
  Load the YOLOv8 model from the specified path.
  """
  print(f"Loading model from {model_path}")
  model = YOLO(model_path)  # Load YOLO model
  return model

# Function to load class labels (from a file or predefined)
def load_classes(classes_file):
  """
  Load class labels from a given file.
  """
  with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
  return classes

# Function to detect characters in the image using YOLOv8
def detect_captcha(model, image):
  """
  Detect characters in the captcha image using the YOLOv8 model.
  """
  print("Processing image with YOLOv8...")

  # Run inference on the image
  results = model.predict(image, conf=0.5, imgsz=512)  # Adjust confidence and image size as necessary

  # Extract bounding box data
  detections = results[0].boxes.data.cpu().numpy()

  # Sort characters by their x-coordinates to preserve order
  characters = []
  for det in detections:
    x1, y1, x2, y2, conf, cls = det
    characters.append((int(cls), (x1, y1, x2, y2)))  # Append class ID and bounding box

  characters.sort(key=lambda char: char[1][0])  # Sort by x1 (horizontal position)

  return characters


# Function to extract text from image
def extract_text_from_image(base64_image, model, class_labels):
  try:
    # Convert the base64 image to bytes
    image_bytes = base64.b64decode(base64_image[22:])
    image_buffer = io.BytesIO(image_bytes)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    current_time = datetime.now().time()  # Get the current system time

    # Define time ranges when gray_scaling should be skipped
    skip_grayscale_ranges = [
      (time(9, 55), time(10, 10)),  # 9:55 AM to 10:10 AM
      (time(10, 55), time(11, 10))  # 10:55 AM to 11:10 AM
    ]
    # Check if the current time falls in any of the defined ranges
    skip_grayscale = any(start <= current_time <= end for start, end in skip_grayscale_ranges)
    # Open the image using PIL (Python Imaging Library)
    image = Image.open(image_buffer)

    if skip_grayscale:
      # Do not grayscale the image, save as is
      print(f"Skipping grayscale due to time: {current_time}")
      processed_image = image
    else:
      # Convert the image to grayscale
      grayscale_image = image.convert("L")
      grayscale_image.save(f"{timestamp}_grayscale.png")
      print(f"grayscale due to time: {current_time}")
      # Convert the grayscale image back to RGB for YOLO processing
      processed_image = Image.merge("RGB", (grayscale_image, grayscale_image, grayscale_image))

    # Resize the image to 640x200
    # resized_image = processed_image.resize((640, 200))
    # print(f"Image resized to 640x200.")

    # Convert the image to RGB if it has an alpha (RGBA) channel
    if processed_image.mode == 'RGBA':
      processed_image = processed_image.convert('RGB')

    # Save the processed image for verification
    processed_image.save(f"{timestamp}_processed.png")

    # Convert PIL image to OpenCV format
    open_cv_image = np.array(processed_image)

    # Detect captcha using YOLO
    detections = detect_captcha(model, open_cv_image)

    # Decode the captcha from detected characters
    decoded_captcha = "".join(class_labels[det[0]] for det in detections)

    return decoded_captcha
  except Exception as e:
    return f"Error processing image: {str(e)}"


# Initialize Flask app
app = Flask(__name__)

#Save Captcha to train the model
def save_captcha_image(base64_image, login_successful, label):
  try:
    # Decode the base64 image
    image_bytes = base64.b64decode(base64_image[22:])
    image_buffer = io.BytesIO(image_bytes)
    image = Image.open(image_buffer)

    # Define image name using a timestamp to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_filename = f"captcha_{timestamp}.png"

    # Save the image in 'train' or 'test' based on login outcome
    if login_successful:
      image_path = os.path.join(TRAIN_DIR, image_filename)
      labels_path = TRAIN_LABELS
    else:
      image_path = os.path.join(TEST_DIR, image_filename)
      labels_path = TEST_LABELS

    # Save the captcha image
    image.save(image_path)

    # Append label to CSV
    with open(labels_path, 'a') as label_file:
      label_file.write(f"{image_filename},{label}\n")

    return True, f"Image saved successfully with login status : {login_successful} @ path: {image_path}"
  except Exception as e:
    return False, f"Error saving image: {str(e)}"

#USING YOLOv8
# Flask endpoint for text extraction
@app.route("/extract-text", methods=["POST"])
def extract_text():
  # Get the base64 image from the request
  data = request.get_json()
  base64_image = data.get("image", "")

  if not base64_image:
    return jsonify({"error": "No base64 image string provided"}), 400

  # Load YOLO model and class labels
  model_path = "yolov8/model/best_10.pt"  # Path to your trained YOLOv8 model
  classes_file = "yolov8/classes.txt"  # Path to your classes file
  model = load_model(model_path)
  class_labels = load_classes(classes_file)

  # Extract text from the base64 image
  extracted_text = extract_text_from_image(base64_image, model, class_labels)

  return jsonify({"extracted_text": extracted_text})

@app.route("/save-image", methods=["POST"])
def save_image():
  # Get the base64 image and login result from the request
  data = request.get_json()
  base64_image = data.get("image", "")
  login_successful = data.get("login_successful", "false").lower() == "true"
  label = data.get("label", "")
  # DEBUG
  # print(f"login_successful: {login_successful} type login_successful: {type(login_successful)}")
  if not base64_image:
    return jsonify({"error": "No base64 image or label string provided"}), 400

  #save the image with provided label and login outcome
  save_status, save_message = save_captcha_image(base64_image, login_successful, label)
  if save_status:
    return jsonify({"status": save_status, "message": save_message}), 200
  else:
    return jsonify({"status": save_status, "message": save_message}), 400

@app.route('/')
def health_check():
  return "Server is running", 200

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Run the OCR extraction server."
  )
  parser.add_argument(
      "--host",
      type=str,
      default="0.0.0.0",
      help="Host address to run the server on (default: 0.0.0.0)",
  )
  parser.add_argument(
      "--port",
      type=int,
      default=5000,
      help="Port to run the server on (default: 5000)",
  )
  args = parser.parse_args()

  # Run Flask server
  app.run(host=args.host, port=args.port)
