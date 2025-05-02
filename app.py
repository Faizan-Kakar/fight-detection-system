import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import streamlit as st
import datetime
import streamlit as st
import cv2
import numpy as np
from PIL import Image  # <-- This is required for Image.fromarray
import time

from utills.frame_to_keypoits import extract_keypoints
from components.alert_box import render_alert_box
from components.incident_cards import render_incident_cards
from components.analytics_panel import render_analytics_panel


from utills.session_utill import initialize_session_state
initialize_session_state()

# Load models
yoloModel = YOLO('yolo11n-pose.pt')
model = load_model('./Models/version3.keras', compile=False)

# Streamlit UI setup
st.set_page_config(page_title="Conflict Alert System", layout="wide")

coll1, coll2 = st.columns([3, 1])
frame_buffer = []

with coll2:   
    alert_box_placeholder = st.empty()  # use this later
    st.markdown("### 🚨 Live Incident Cards")
    incidentsPlaceHolder = st.empty()

with coll1:
    camera_feed = st.empty()
    # Add Start/Stop buttons below video stream
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶️ Start Camera"):
            st.session_state.start_camera = True
    with col_stop:
        if st.button("⛔ Stop Camera"):
            st.session_state.start_camera = False
            
    # Analytical portion is displayed here
    st.markdown("### Analytics Overview")
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        # st.metric("Total Incidents", st.session_state.incident_count, "+15%")
        incident_placeholder = st.empty()
        incident_placeholder.metric("Total Incidents", st.session_state.incident_count, "+15%")
    with col_b:
        st.metric("Detection Accuracy", "98%", "⬆ 2.3%")
    with col_c:
        st.metric("Response Time", "45s", "⬇ 5s faster")
    with col_d:
        # st.metric("Current Status", "No Fight")
        status_placeholder = st.empty()
        status_placeholder.markdown(
            "<h4> <span style='color: green;'>🟢 No Fight</span></h4>",
             unsafe_allow_html=True)
        
    if st.session_state.start_camera:
        cap = cv2.VideoCapture(1)
        cooldown = 5  # seconds
        while st.session_state.start_camera and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Extract Keypoints for each frame
            keypoints, result = extract_keypoints(frame)
            annotated_frame = result.plot()

            if keypoints.shape[0] == 153:
                frame_buffer.append(keypoints)
                if len(frame_buffer) > 41:
                    frame_buffer.pop(0)

                if len(frame_buffer) == 41:
                    input_data = np.expand_dims(frame_buffer, axis=0)
                    prediction = model.predict(input_data)[0][0]
                    label = "🔴 Fighting!" if prediction > 0.5 else "🟢 No Fight"
                    color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)
                    cv2.putText(annotated_frame, f"{label} ({prediction:.2f})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    # label_placeholder.markdown(f"### Status: **{label}**")
                    status_placeholder.markdown(f"### **{label}**")
                    
                    current_time = time.time()
                    if prediction > 0.5 and (current_time - st.session_state.last_incident_time > cooldown):
                        st.session_state.incident_count += 1
                        st.session_state.color = 'red'
                        # Append new incident info
                        incident_info = {
                            "location": "Numl Ghazali",  # You can update dynamically
                            "cam_id": "#3459",
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "Unattended"
                        }
                        st.session_state.incidents.append(incident_info)
                        
                        # Render Incident Cards
                        incidentsPlaceHolder.markdown(render_incident_cards(st.session_state.incidents), unsafe_allow_html=True)
                        
                        st.session_state.last_incident_time = current_time
                        incident_placeholder.metric("Total Incidents", st.session_state.incident_count, "+15%")
                        
                    if prediction < 0.5 and (current_time - st.session_state.last_incident_time > cooldown):
                           st.session_state.color = 'green'                         
                    # For Changing the Status
                    status_placeholder.markdown(
                        f"<h4><span style='color: {color};'>{label}</span></h4>",
                        unsafe_allow_html=True)
                    
            # Displaying the alert box        
            alert_box_placeholder.markdown(render_alert_box(st.session_state.color), unsafe_allow_html=True)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            # camera_feed.image(img_pil, use_container_width=True)
            img_resized = img_pil.resize((600, 300))  # (width, height)
            camera_feed.image(img_resized)  # or any pixel width you prefer
            time.sleep(0.01)

        cap.release()
        camera_feed.empty()
        # label_placeholder.empty()


    
    
