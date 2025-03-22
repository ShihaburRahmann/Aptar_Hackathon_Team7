import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO  
from PIL import Image

model = YOLO("best.pt")  

CONF_THRESH = 0.85  
X_THRESHOLD = 100  # Adjust for grouping columns

st.set_page_config(page_title="Box Detection System", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #000000;
            color: white;
        }
        .header {
            text-align: center;
            color: white;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 30px;
            margin-bottom: 20px;
        }
        .logo {
            width: 100px;
            height: auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='header'>📦 Box Detection System - Team 7</h1>", unsafe_allow_html=True)


st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_cv2 = np.array(image)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    image_yolo = image_cv2.copy()
    results = model(image_yolo)

    boxes = []
    for result in results:
        for box in result.boxes:
            if hasattr(box, "conf") and box.conf[0] >= CONF_THRESH:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda b: b[0])
    columns = []
    for box in boxes:
        placed = False
        for column in columns:
            if abs(column[-1][0] - box[0]) < X_THRESHOLD:
                column.append(box)
                placed = True
                break
        if not placed:
            columns.append([box])

    column_counts = [len(col) for col in columns]

    for col_idx, column in enumerate(columns):
        if len(column) > 5:
            topmost = min(column, key=lambda b: b[1])
            bottommost = max(column, key=lambda b: b[3])
            x1, y1, x2, y2 = topmost[0], topmost[1], bottommost[2], bottommost[3]
            cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 0, 255), 5)
            x_mid = (x1 + x2) // 2
            cv2.putText(image_cv2, f"{len(column)}", (x_mid, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    image_output = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(image_output, caption="Processed Image", use_container_width=True)

st.markdown(
    """
    <hr>
    <div style='text-align: center; color: white;'>
        <p><b>Developed by Team 7</b> | Powered by YOLO</p>
        <p>Alexis Gbeckor-Kove | Ishara Galbokka Hewage | Puja Dhakal | Saad Abdullah | Sarosh Krishan | Shihabur Samrat </p>
    </div>
    """,
    unsafe_allow_html=True
)