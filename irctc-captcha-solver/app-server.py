import argparse
import os
import numpy as np
from PIL import Image
import io
import base64
import easyocr
from flask import Flask, request, jsonify
from datetime import datetime
from PIL import Image
import pytesseract
# Specify the exact path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# DataSet Directory
TRAIN_DIR = "datasets/train/images"
TEST_DIR = "datasets/test/images"
TRAIN_LABELS = "datasets/train/labels.csv"
TEST_LABELS = "datasets/test/labels.csv"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
# Initialize EasyOCR Reader
reader = easyocr.Reader(["en"], model_storage_directory="./EasyOCR")

# Initialize Flask app
app = Flask(__name__)
# Using pytesseract
def extract_text_from_image(base64_image):
  try:
    # Convert the base64 image to bytes
    image_bytes = base64.b64decode(base64_image[22:])

    # Create a BytesIO object from the image bytes
    image_buffer = io.BytesIO(image_bytes)

    # Open the image using PIL (Python Imaging Library)
    image = Image.open(image_buffer)

    # Convert the image to grayscale (optional but can improve OCR accuracy)
    image = image.convert("L")

    # Use Tesseract to extract text from the image
    result = pytesseract.image_to_string(image, config='--psm 7')  # --psm 7 assumes a single line of text

    # Process the result if any text is found
    if result.strip():
      return result.strip().replace(" ", "")
    else:
      return "ABCDEF"
  except Exception as e:
    return f"Error processing image: {str(e)}"
# def extract_text_from_image(base64_image):
#   try:
#     # Convert the base64 image to bytes
#     image_bytes = base64.b64decode(base64_image[22:])
#     # Create a BytesIO object from the image bytes
#     image_buffer = io.BytesIO(image_bytes)
#     # Open the image using PIL (Python Imaging Library)
#     image = Image.open(image_buffer)
#     # Convert the image to grayscale (optional but can improve OCR accuracy)
#     image = image.convert("L")
#
#     # Convert PIL image to OpenCV format
#     open_cv_image = np.array(image)
#
#     result = reader.readtext(open_cv_image, detail=0)
#     if result:
#       return result[0].replace(" ", "")
#     else:
#       return "ABCDEF"
#   except Exception as e:
#     return f"Error processing image: {str(e)}"
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

@app.route("/extract-text", methods=["POST"])
def extract_text():
  # Get the base64 image from the request
  data = request.get_json()
  base64_image = data.get("image", "")

  if not base64_image:
    return jsonify({"error": "No base64 image string provided"}), 400

  extracted_text = extract_text_from_image(base64_image)
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
