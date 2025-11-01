import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# ============================================================
# ✅ MODEL CLASS (MATCHING TRAINING VERSION)
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
# ✅ IMAGE PREPROCESSING PIPELINE
# ============================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ============================================================
# ✅ STREAMLIT APP UI
# ============================================================
st.title("🎵 SimpleCNN Mel Spectrogram Classifier")
st.write("Upload a **mel spectrogram image** (128×128, grayscale) to predict its class.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Mel Spectrogram", use_column_width=True)

    # Preprocess image
    input_tensor = transform(img).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())

    # Display results
    st.success(f"**Prediction: Class {pred}**")
    st.write(f"**Confidence: {probs[pred]:.1%}**")

    # Bar chart
    class_names = [f"Class {i}" for i in range(len(probs))]
    st.bar_chart(dict(zip(class_names, probs)))

# ============================================================
# ✅ FOOTER
# ============================================================
st.caption("Model: SimpleCNN | Framework: PyTorch | App powered by Streamlit 🚀")
