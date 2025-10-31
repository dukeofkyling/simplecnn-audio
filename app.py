import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# === YOUR MODEL CLASS (COPY FROM YOUR NOTEBOOK) ===
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    model = SimpleCNN(n_classes=2)  # ← Change n_classes if not 2
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# === PREPROCESSING ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# === UI ===
st.title("SimpleCNN Mel Spectrogram Classifier")
st.write("Upload a **mel spectrogram image** (128×128, grayscale)")

uploaded_file = st.file_uploader("Choose image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Mel Spectrogram", use_column_width=True)
    
    # Preprocess
    input_tensor = transform(img).unsqueeze(0)  # (1, 1, 128, 128)
    
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
