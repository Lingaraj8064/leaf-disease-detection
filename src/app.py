import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

# Set the correct template folder
app = Flask(__name__, template_folder="templates")

# Set upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = "models/leaf_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (update according to your dataset)
CLASS_LABELS = ["Healthy", "Diseased"]

# Image preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            file = request.files.get("file")
            print("📂 File received:", file)  # Debugging

            if not file:
                print("❌ No file uploaded")
                return jsonify({"error": "No file uploaded"}), 400

            filename = file.filename.replace(" ", "_").lower()
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            print(f"✅ Image saved at: {file_path}")

            # Preprocess image
            img_array = preprocess_image(file_path)
            print(f"✅ Image processed: Shape - {img_array.shape}")

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = CLASS_LABELS[int(prediction[0] > 0.5)]
            print(f"✅ Prediction: {predicted_class}")

            return jsonify({"image_url": f"/{file_path}", "prediction": predicted_class})

        except Exception as e:
            print(f"❌ Error: {e}")  # Log error message
            return jsonify({"error": str(e)}), 500

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
