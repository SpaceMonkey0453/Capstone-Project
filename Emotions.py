import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model as tf_load_model

def load_model():
    model = tf_load_model("Emotions.h5")
    return model
# Load the emotion tracking model


# Define a function to preprocess the input image
def preprocess_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize the image to 48x48 pixels
    resized = cv2.resize(gray, (48, 48), interpolation = cv2.INTER_AREA)
    # Reshape the image to a 4D tensor with a single channel
    tensor = resized.reshape(1, 48, 48, 1)
    # Normalize the pixel values to be between 0 and 1
    tensor = tensor.astype('float32') / 255.0
    return tensor

def get_emoji(emotion):
    emoji_dict = {
        'Angry': '😠',
        'Disgust': '🤢',
        'Fear': '😨',
        'Happy': '😄',
        'Sad': '😔',
        'Surprise': '😲',
        'Neutral': '😐'
    }
    return emoji_dict[emotion]

# Define the Streamlit app
def app():
    st.title("Facial Emotion Tracking App")
    # Load the emotion tracking model
    model = load_model()
    # Allow the user to upload an image file
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the image from the uploaded file
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Find the faces in the image using OpenCV face detection
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # If a face is detected, crop the image to the face region and preprocess it for emotion tracking
        if len(faces) > 0:
            # Find the largest face by area
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Crop the image to the face region
            cropped = img[y:y+h, x:x+w]
            
            # Preprocess the cropped image for emotion tracking
            tensor = preprocess_image(cropped)
            
            # Make a prediction on the preprocessed image using the emotion tracking model
            prediction = model.predict(tensor)
            
            # Map the prediction to a human-readable emotion label
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            predicted_label = emotion_labels[np.argmax(prediction)]
            emoji = get_emoji(predicted_label)
            
            # Display the original image with the detected face region and the predicted emotion label
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
            st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), caption="Cropped Face Region", use_column_width=True)
            
            with st.container():
                st.write(f"Predicted Emotion: <span style='font-size: 50px; font-weight: bold;'>{predicted_label} {emoji}</span>", 
                         unsafe_allow_html=True, 
                         key='predicted_emotion',
                         )
                st.markdown("<style>.css-1aumxhk{text-align:center !important}</style>", unsafe_allow_html=True)
        
        # If no face is detected, display an error message
        else:
            st.error("No face detected in the uploaded image.")

if __name__ == "__main__":
    app()