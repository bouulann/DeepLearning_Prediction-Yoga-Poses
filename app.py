from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
MODEL_PATH = 'PoseYoga.h5'  # Replace with your .h5 file path
model = load_model(MODEL_PATH)

# Define class names
class_names = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]  # Replace with your actual class names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess the image
            img = load_img(filepath, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            predicted_label = class_names[predicted_class]
            confidence = prediction[0][predicted_class] * 100

            # Clean up the uploaded file
            os.remove(filepath)

            # return render_template('index.html', prediction={
            #     "predicted_label": predicted_label,
            #     "confidence": f"{confidence:.2f}"
            # })
            return render_template('index.html',
                                    predicted_label=predicted_label,
                                      confidence=f"{confidence:.2f}")


        except Exception as e:
            os.remove(filepath)
            return render_template('index.html', error=str(e))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
