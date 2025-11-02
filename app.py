# ===========================
# Streamlit App: Respiratory Sound Classifier
# ===========================
import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import librosa
import os
import requests

# ---------------------------
# Model definition
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=4, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------------
# Load model safely
# ---------------------------
MODEL_URL = "https://your-model-link/model.pth"  # 🔗 Replace this with your hosted model link

@st.cache_resource
def load_model(model_path="model.pth", n_classes=4):
    model = SimpleCNN(n_classes=n_classes)
    if not os.path.exists(model_path):
        st.info("⬇️ Downloading model file...")
        try:
            r = requests.get(MODEL_URL)
            with open(model_path, "wb") as f:
                f.write(r.content)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return model

    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.warning(f"⚠️ Model load issue: {e}")
    model.eval()
    return model

# ---------------------------
# Audio preprocessing
# ---------------------------
def audio_to_mel_tensor(file_bytes, n_mels=128):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())  # normalize 0–1
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()  # shape (1,1,128,128)
    return mel_tensor

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎵 Respiratory Sound Classifier")
st.write("Upload a **breathing sound audio file** (`wav`, `mp3`, or `ogg`) to detect:")
st.markdown("**Normal**, **Wheezes**, **Crackles**, or **Both (Crackles + Wheezes)**")

uploaded_file = st.file_uploader("Choose audio file", type=["wav", "mp3", "ogg"])
if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    # Convert audio to mel spectrogram
    mel_tensor = audio_to_mel_tensor(uploaded_file.read())

    # Load model (4 classes)
    n_classes = 4
    labels = ["Normal", "Wheezes", "Crackles", "Both"]
    model = load_model("model.pth", n_classes=n_classes)

    # Predict
    with torch.no_grad():
        output = model(mel_tensor)
        probs = torch.softmax(output, dim=1).squeeze().numpy()
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]

    st.success(f"**Prediction: {pred_label}**")
    st.write(f"**Confidence: {probs[pred_idx]:.1%}**")
    st.bar_chart(dict(zip(labels, probs)))

st.caption("Model: SimpleCNN | Framework: PyTorch | App powered by Streamlit 🚀")
