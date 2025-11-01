import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import tempfile

# =========================
# ✅ Model Class (SimpleCNN)
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=4, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =========================
# ✅ Load Trained Model
# =========================
@st.cache_resource
def load_model():
    model = SimpleCNN(n_classes=4)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# =========================
# ✅ Streamlit UI
# =========================
st.title("🎵 Patient Audio Classifier")
st.write("Upload an audio file (wav/mp3) to predict condition: Normal, Wheezles, Crackles, or Both")

uploaded_file = st.file_uploader("Choose audio file", type=["wav","mp3"])

if uploaded_file is not None:
    # Save uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Play the audio
    st.audio(temp_path)

    # Convert audio to mel spectrogram
    y, sr = librosa.load(temp_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    mel_db = np.expand_dims(mel_db, axis=0)  # channel
    mel_db = np.expand_dims(mel_db, axis=0)  # batch
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        output = model(mel_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred_class = int(probs.argmax())

    # Map prediction to class name
    class_map = {0: "Normal", 1: "Wheezles", 2: "Crackles", 3: "Both"}
    condition = class_map.get(pred_class, "Unknown")

    # Display results
    st.success(f"Predicted condition: {condition}")
    st.write(f"Confidence: {probs[pred_class]:.1%}")

    # Show all class probabilities
    st.bar_chart(dict(zip([class_map[i] for i in range(len(probs))], probs)))

st.caption("Model: SimpleCNN | Framework: PyTorch | App powered by Streamlit 🚀")
