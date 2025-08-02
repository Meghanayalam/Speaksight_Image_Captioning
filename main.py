import streamlit as st
from keras.models import load_model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from PIL import Image
import os
from gtts import gTTS
from io import BytesIO
import tempfile

# Load model and tokenizer
model = load_model("liga_30Kv2.h5")
with open("word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)
with open("idx_to_word.pkl", "rb") as f:
    idx_to_word = pickle.load(f)

vocab_size = len(word_to_idx) + 1
max_length = 34  # Use the max_length from your training

# Load ResNet model
resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Caption generation function
def extract_features(img_path):
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = resnet.predict(image, verbose=0)
    return feature

def generate_caption(photo):
    in_text = "startseq"
    for i in range(max_length):
        sequence = [word_to_idx.get(w, 0) for w in in_text.split()]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([photo, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = idx_to_word.get(y_pred)
        if word is None or word == "endseq":
            break
        in_text += " " + word
    return in_text.replace("startseq", "").strip()

# Text-to-speech
def speak(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        return tmp_file.name

# Streamlit UI
st.set_page_config(page_title="SpeakSight - Image Captioning", layout="centered")
st.title("🧠 SpeakSight")
st.write("Upload an image and let AI describe it out loud.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save to disk
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    img.save(temp_path)

    # Generate caption
    photo = extract_features(temp_path)
    caption = generate_caption(photo)

    st.success("📝 Caption:")
    st.markdown(f"**{caption}**")

    # Speak it
    st.audio(speak(caption), format="audio/mp3")

    # Cleanup
    os.remove(temp_path)
