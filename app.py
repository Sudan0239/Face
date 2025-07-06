from flask import Flask, render_template, request, jsonify
import cv2
from deepface import DeepFace
import numpy as np
import base64
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Age correction factor
AGE_CORRECTION = 5

# Function to get age range
def get_age_range(age):
    return f"{max(0, age - 5)}-{age + 5}"

def analyze_face(image_data):
    try:
        # Decode the base64 image
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Analyze the face using VGG-Face model and Keras backend
        result = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion', 'race'],
                                  enforce_detection=False, detector_backend='opencv',
                                  align=True, silent=True)

        # Extract the results
        age = max(0, result[0]['age'] - AGE_CORRECTION)  # Apply correction factor
        age_range = get_age_range(age)
        gender = result[0]['dominant_gender']
        emotion = result[0]['dominant_emotion']
        race = result[0]['dominant_race']

        return {
            'age': age,
            'age_range': age_range,
            'gender': gender,
            'emotion': emotion,
            'race': race
        }
    except Exception as e:
        app.logger.error(f"Error in analyze_face: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        image_data = request.json['image']
        app.logger.info(f"Received image data of length: {len(image_data)}")
        
        result = analyze_face(image_data)
        if result:
            app.logger.info(f"Analysis result: {result}")
            return jsonify(result)
        else:
            app.logger.warning("No face detected or analysis failed")
            return jsonify({'error': 'No face detected or analysis failed'}), 400
    except Exception as e:
        app.logger.error(f"Error in /analyze route: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)