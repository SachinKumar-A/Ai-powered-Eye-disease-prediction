import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd
import base64  # For encoding background image
import tensorflow as tf

# Function to encode the image
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Path to your background image (Ensure this is correct)
background_image_path = r'C:\Users\sksan\OneDrive\Desktop\infosys intern\New folder (2)\ss.jpg'  # Change to your correct path

# Generate base64 string
base64_bg = get_base64_image(background_image_path)

# Apply CSS to set the background
# Apply CSS to set the background and text color
page_bg_css = f"""
<style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_bg}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #FFFFFF;  /* Light Cyan Text */
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: #FFFFFF !important;  /* Ensuring headings also change */
    }}

    .stSidebar {{
        color: #FFFFFF !important;  /* Sidebar text */
    }}

    .stButton>button {{
        color: black !important;
        background-color: #FFFFFF !important; /* Light Cyan Buttons */
    }}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)


# Define class names
CLASS_NAMES = [
    'Age-Related Macular Degeneration', 'Branch Retinal Vein Occlusion', 'Cataract', 'Diabetic Retinopathy', 'Drusen', 'Glaucoma', 'Hypertension', 'Media Haze', 'Normal', 'Others', 'Pathological Myopia', 'Tessellation'
]

IMG_SIZE = (224, 224)

st.title("MediScan: AI-Powered Medical Image Analysis for Disease Diagnosis")
st.write("Upload eye images to predict the condition, whether it's a disease or normal.")

# Cache model loading
@st.cache_resource
def load_trained_model():
    try:
        return load_model(r'C:\Users\sksan\OneDrive\Desktop\infosys intern\New folder (2)\model231.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

with st.spinner('Loading model...'):
    model = load_trained_model()

if not model:
    st.stop()

# Initialize session state
if 'patients' not in st.session_state:
    st.session_state.patients = []

# Sidebar: Patient information
st.sidebar.header("Patient Information")
patient_id = st.sidebar.text_input("Patient ID")
patient_name = st.sidebar.text_input("Patient Name")
patient_age = st.sidebar.number_input("Patient Age", min_value=0, max_value=120, step=1)

# File uploader
uploaded_file = st.file_uploader("Choose an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")  # Convert to RGB
    image = image.resize(IMG_SIZE)  # Resize to match model input
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize between 0-1
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def temperature_scaled_softmax(logits, temp=0.5):
    """ Apply temperature scaling to softmax output """
    exp_scaled = np.exp(logits / temp)
    return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=False, width=300)

    if st.button("Predict & Save Data"):
        img_array = preprocess_image(image)

        # Perform prediction
        raw_predictions = model.predict(img_array)  # Get logits (raw scores)
        adjusted_predictions = temperature_scaled_softmax(raw_predictions, temp=0.5)  # Apply temperature scaling

        predicted_class_index = np.argmax(adjusted_predictions, axis=1)[0]
        predicted_label = CLASS_NAMES[predicted_class_index]
        confidence = np.max(adjusted_predictions) * 100  # Confidence score

        # Debugging: Show probability distribution
        st.write("Adjusted Class Probabilities:", {CLASS_NAMES[i]: round(prob * 100, 2) for i, prob in enumerate(adjusted_predictions[0])})

        # Save patient data
        st.session_state.patients.append({
            "Patient ID": patient_id,
            "Patient Name": patient_name,
            "Patient Age": patient_age,
            "Eye Condition": predicted_label.upper(),
            "Confidence (%)": round(confidence, 2)
        })

        st.success(f"Prediction: **{predicted_label}** with {confidence:.2f}% confidence")

# Button to clear all fields
if st.button("Clear All Fields"):
    st.session_state.patients.clear()

# Display patient data
if st.session_state.patients:
    st.write("## Prediction and Patient Information:")
    result_df = pd.DataFrame(st.session_state.patients)
    st.table(result_df)

    # Download predictions as CSV
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", data=csv, file_name='predictions.csv', mime='text/csv')
