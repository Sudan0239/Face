import cv2
from deepface import DeepFace
import numpy as np

# Age correction factor
AGE_CORRECTION = 5

# Function to get age range
def get_age_range(age):
    return f"{max(0, age - 5)}-{age + 5}"

def analyze_face(frame):
    try:
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
        
        return f"Age: {age} ({age_range}), Gender: {gender}, Emotion: {emotion}, Race: {race}"
    except Exception as e:
        return "No face detected"

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze the face
        result = analyze_face(frame)

        # Display the result on the frame
        cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Face Characteristics', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()