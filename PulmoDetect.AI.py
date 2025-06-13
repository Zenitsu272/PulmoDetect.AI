import streamlit as st
import requests
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("xray_lung_classifier_model.h5")

model = load_model()
class_names = ['lung_opacity', 'normal', 'pneumonia']
img_size = (224, 224)

# Initialize session variable
if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None

# App layout
st.title(" PulmoDetect.AI - The Ultimate Lung Disease Classifier!")
if st.button("About PulmoDetect.AI"):
    st.success("PulmoPredict.AI is an intelligent lung disease prediction tool designed based on CNN models to assist in early detection of common pulmonary conditions\n using chest X-ray images. Powered by machine learning, the app analyzes uploaded scans and provides fast, accurate predictions\n for diseases like Pneumonia, Lung Opacity, and Normal condition, helping users and healthcare providers make informed decisions.With a user-friendly interface and reliable insights,\n PulmoPredict brings the power of AI to respiratory health — anytime, anywhere.")


tab1, tab2, tab3 = st.tabs([
    "Disease Predictor + Patient Report",
    "Health Assistant (MRI only)",
    "About this App"
])

with tab1:
    st.header("Upload Chest X-ray for Prediction")
    uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_data = Image.open(uploaded_file).convert("RGB")
        st.image(image_data, caption="Uploaded X-ray", use_column_width=True)

        # Preprocess
        img_resized = image_data.resize(img_size)
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Store in session state for Tab 2
        st.session_state.predicted_class = predicted_class

        # Show result
        st.success(f"🩺 Predicted Class: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

with tab2:
    st.header("Healthcare Assistant : Ask-Analyse-Act")
    st.error("""This is your ultimate disease reviewer AI : Ask-Analyse-Act.
    Ask questions related to the predicted lung disease and get helpful insights to understand 
    and analyze your condition better. You can also ask general queries about lung diseases -- symptoms, causes,
    diagnosis, treatment options, or prevention tips. Powered by Gemini AI,this assistant
    is here to support you with accurate, reliable, and easy-to-understand medical information.""")
    
    predicted = st.session_state.get("predicted_class", "an unspecified condition")

    prompt = st.text_input("Ask a question about lung diseases:")

    if prompt:
        API_KEY = st.secrets["GEMINI_API_KEY"]
        MODEL_NAME = "models/gemini-1.5-flash"
        URL = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent?key={API_KEY}"

        headers = {"Content-Type": "application/json"}

        # Compose dynamic prompt
        text_prompt = (
            f"You are a professional healthcare assistant trained to answer questions related to lung diseases only. "
            f"These include pneumonia, lung opacity (such as seen in X-rays), and normal lung conditions. "
            f"The user has recently been predicted to have **{predicted}** from a chest X-ray scan. "
            f"Start your response by briefly explaining what **{predicted}** means and then answer their question. "
            f"If the question is outside the scope of lung diseases, kindly respond that you're limited to lung health only. "
            f"If asked, mention that this app was developed by Vaishak. "
            f"\n\nUser Question: {prompt}"
            )


        data = {"contents": [{"parts": [{"text": text_prompt}]}]}
        response = requests.post(URL, headers=headers, json=data)

        if response.status_code == 200:
            reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            st.write(reply)
        else:
            st.error(" Gemini API Error: " + response.text)

with tab3:
    st.header("About This App ")
    st.markdown("[Visit LinkedIn](www.linkedin.com/in/vaishak-t-96b93930a)")

    #st.link_button("Visit LinkedIn", "www.linkedin.com/in/vaishak-t-96b93930a")

    st.markdown("**The app uses a self-made Deep learning model to predict the diseases. The results may not be 100% accurate.\n Dvelopments in the model is been made regularly.**")
    prod = st.selectbox("Select the details you wanna view ",["Uses of the App","Deep learning model details","Tech stack"])
    if prod=="Uses of the App":
        st.markdown("""
         **Early Detection of Lung Diseases** ""
        Helps in the early identification of diseases such as Pneumonia, Tuberculosis, and Lung Cancer using chest X-ray images, allowing timely medical intervention.

         **AI-Powered Diagnosis Support** :
        Assists doctors and healthcare workers by providing AI-generated predictions to complement clinical judgment and reduce diagnostic errors.
    
         **Smart Disease Query Assistant** :
        Integrated with Gemini AI, the app allows users to ask questions about their predicted condition and receive detailed, medically relevant answers instantly.

         **User-Friendly Image Upload** :
        Enables users to easily upload their chest X-ray and receive a quick diagnosis without any complex steps or prior medical knowledge.

          **Remote Health Screening** :
        Useful in rural or underserved areas where access to radiologists is limited, providing a fast and accessible alternative for preliminary screening.
    
         **Continuous Improvement** : 
        As better deep learning models are developed and integrated, prediction accuracy will improve — ensuring users always get the best available analysis.""")

        
    elif prod=="Deep learning model details":
        st.success("This deep learning model classifies chest X-ray images into lung disease categories using a CNN. ")
        st.warning("Images are resized to 224x224 and processed in batches of 32. ")
        st.success("The model has 3 convolutional layers (with 32, 64, 128 filters, 3x3 kernels) using ReLU activation, ")
        st.warning("Each followed by a MaxPooling layer to reduce spatial dimensions.")
        st.success("It ends with a dense layer and softmax activation for multiclass prediction. ")
        st.warning("Regularization techniques like L2 and Dropout are used to avoid overfitting. ")
        st.success("Training is optimized with an EarlyStopping callback to prevent unnecessary epochs. ")
        
        
       
    elif prod=="Tech stack":
        st.success("Tensorflow")
        st.warning("Keras")
        st.success("Streamlit")
        st.warning("GeminiAPI - Gemini-1.5-flash ")
        st.success("Python")
        
        


