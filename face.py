import cv2
import joblib
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image

# Load the trained classifier and label encoder
clf = joblib.load('arcface_classifier.pkl')
le = joblib.load('label_encoder.pkl')

# Initialize the FaceAnalysis app
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB format and then to a NumPy array
    img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = app.get(img_array)

    for face in faces:
        # Extract the embedding
        embedding = face.normed_embedding.reshape(1, -1)

        # Predict the user ID
        probabilities = clf.predict_proba(embedding)
        max_prob_idx = np.argmax(probabilities)
        max_prob = probabilities[0, max_prob_idx]

        # Set a threshold for recognition
        threshold = 0.9# You can adjust this value based on your requirements

        if max_prob > threshold:
            user_id = le.inverse_transform([max_prob_idx])[0]
            label = f"ID: {user_id} ({max_prob:.2f})"
        else:
            label = "No match"

        # Draw a rectangle around the face and label it
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
