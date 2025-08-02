# Speaksight_Image_Captioning
SpeakSight is an AI-powered application that generates and vocalizes descriptive captions for images using a deep learning pipeline with ResNet50, a custom Keras model, and text-to-speech technology.

# Speaksight: Deep Learning–Powered Image Captioning System

Speaksight is an AI-powered image captioning system that generates natural language descriptions for images using a convolutional-recurrent architecture. The project is built with a focus on accuracy, clarity, and accessibility. It integrates computer vision and natural language processing to bridge the gap between seeing and describing.

## 📌 Overview

Image captioning is the task of generating textual descriptions for visual content. Speaksight approaches this by using a pre-trained CNN to understand image content and an LSTM-based RNN to generate fluent, descriptive captions.

This system includes:
- A Streamlit-powered interactive frontend
- A trained encoder-decoder model
- Optional text-to-speech integration for accessibility

## 🧠 Architecture

**1. Encoder**  
A pre-trained CNN (e.g., InceptionV3) extracts spatial image features and compresses them into a high-level vector representation.

**2. Decoder**  
An RNN with LSTM units decodes the visual feature vector into a sequence of words using learned embeddings and vocabulary mappings.

**3. Output**  
The model produces a complete sentence describing the image, optionally converted to speech using gTTS.

**Workflow Summary**:
Image Upload → Preprocessing → Feature Extraction → Caption Generation → (Optional) Text-to-Speech

## 🧰 Technology Stack

- **TensorFlow/Keras**: Deep learning model architecture
- **Streamlit**: Interactive user interface
- **Pillow (PIL)**: Image loading and preprocessing
- **gTTS**: Text-to-speech conversion
- **Pickle**: Serialization of token mappings
- **NumPy**: Numerical computations

## 📂 Project Structure

```
speaksight/
├── main.py                 # Streamlit app logic
├── requirements.txt        # Project dependencies
├── liga_30Kv2.h5           # Trained captioning model weights (confidential)
├── word_to_idx.pkl         # Word-to-index mapping dictionary (confidential)
├── idx_to_word.pkl         # Index-to-word mapping dictionary (confidential)
├── README.md               # Project documentation
```

## 📄 File Descriptions

- `main.py`: The primary script that launches the Streamlit app, loads the model, handles image input, and displays generated captions.
- `requirements.txt`: Contains a list of all Python packages needed to run the project.
- `liga_30Kv2.h5`: Trained model weights. This file is confidential and not distributed publicly.
- `word_to_idx.pkl` & `idx_to_word.pkl`: Python dictionaries used to map vocabulary to model-compatible indices and vice versa.

## 💻 Installation

Ensure you have Python 3.8+ and `pip` installed. Then follow these steps:

### 1. Clone the Repository

```
git clone https://github.com/Meghanayalam/Speaksight_Image_Captioning.git
cd Speaksight_Image_Captioning
```

### 2. Install Required Packages

```
pip install -r requirements.txt
```

### 3. Run the Application

```
streamlit run main.py
```

After launching, your browser will open a local app where you can upload images and generate captions.


## 🔊 Text-to-Speech (Optional)

If enabled in the UI, captions can be spoken aloud using Google’s Text-to-Speech engine (gTTS). This requires an active internet connection.

## 🔧 Future Improvements

- Integrate attention mechanisms (Bahdanau or Transformer-based)
- Upgrade decoder to use Transformers (e.g., BERT + Vision encoders)
- Dockerize the app for containerized deployment
- Deploy on Hugging Face Spaces or AWS Lambda
- Add multilingual captioning support
- Caption videos via frame sampling

## 🙋‍♀️ About the Author

**Meghana Yalam**  
Deep Learning Developer & NLP Enthusiast  
GitHub: [https://github.com/Meghanayalam](https://github.com/Meghanayalam)

## 📄 License

This project is intended for academic and research use only.  
The models and training data are confidential and are not distributed for public or commercial use.


