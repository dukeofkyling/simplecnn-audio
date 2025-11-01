import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io

# ============================================================
# ✅ MODEL CLASS
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=2, in_ch=1):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
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


# ============================================================
# ✅ LOAD TRAINED MODEL
# ============================================================
@st.cache_resource
def load_model():
    model = SimpleCNN(n_classes=2)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ============================================================
# ✅ AUDIO FILE UPLOAD AND MEL SPECTROGRAM CONVERSION
# ============================================================
def audio_to_mel_image(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(2, 2))
    librosa.display.specshow(mel_db, sr=sr, x_axis=None, y_axis=None, ax=ax, cmap="gray_r")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ============================================================
# ✅ STREAMLIT UI
# ============================================================
st.title("🎧 SimpleCNN Audio Classifier (Mel Spectrograms)")
st.write("Upload a **.wav** file — it will be converted into a mel spectrogram automatically.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")

    # Convert to mel spectrogram
    mel_img = audio_to_mel_image(audio_bytes)
    st.image(mel_img, caption="Generated Mel Spectrogram", use_column_width=True)

    # Preprocess for model
    input_tensor = transform(mel_img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())

    st.success(f"**Prediction: Class {pred}**")
    st.write(f"**Confidence: {probs[pred]:.1%}**")

    # Bar chart
    class_names = [f"Class {i}" for i in range(len(probs))]
    st.bar_chart(dict(zip(class_names, probs)))

st.caption("Model: SimpleCNN | Trained on Mel Spectrograms | App powered by Streamlit 🚀")
