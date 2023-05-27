import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np

# Load the H5 file
model = load_model('F:\\download\\best_model.h5')

# Load the face cascade classifier XML file
face_cascade = cv2.CascadeClassifier(r"C:\Users\User\Desktop\haarcascade_frontalface_default.xml")

# Define the emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Create a video capture object
cap = cv2.VideoCapture(0)

# Loop until the user presses ESC
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(grayscale_frame, scaleFactor=1.1, minNeighbors=5)

    # Loop over the faces
    for face in faces:
        # Crop the face from the frame
        face_image = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]

        # Convert the face image to grayscale
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        # Resize the face image to 48x48 pixels
        face_image = cv2.resize(face_image, (48, 48))

        # Convert the face image to an array
        face_image = tf.keras.preprocessing.image.img_to_array(face_image)

        # Normalize the face image
        face_image = face_image / 255.0

        # Reshape the face image
        face_image = np.expand_dims(face_image, axis=0)

        # Predict the emotion of the face
        prediction = model.predict(face_image)

        # Check if prediction array is not empty
        if len(prediction[0]) > 0:
            # Get the emotion with the highest probability
            emotion = emotions[np.argmax(prediction[0])]
        else:
            emotion = "Unknown"

        # Draw a rectangle around the face
        cv2.rectangle(frame, face, (0, 0, 255), 2)

        # Write the emotion on the frame
        cv2.putText(frame, emotion, (face[0], face[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If the user presses ESC, stop the loop
    if key == 27:
        break

# Release the video capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
