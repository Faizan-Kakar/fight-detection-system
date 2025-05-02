from tensorflow.keras.models import load_model
from ultralytics import YOLO
import numpy as np


# Load models
yoloModel = YOLO('yolo11n-pose.pt')
model = load_model('./Models/version3.keras', compile=False)

# Keypoints Extracting using YOLO
def extract_keypoints(frame):
    max_persons = 3
    num_keypoints = 17
    features_per_person = num_keypoints * 3
    results = yoloModel.predict(frame)
    result = results[0]
    frame_keypoints = []
    for result in results:
        keypoints = result.keypoints.data.cpu().numpy()
        for person_keypoints in keypoints:
            if person_keypoints.shape[0] != 17:
                continue
            flattened = person_keypoints.flatten()
            frame_keypoints.append(flattened)

    while len(frame_keypoints) < max_persons:
        frame_keypoints.append(np.zeros(features_per_person))

    frame_keypoints = frame_keypoints[:max_persons]
    flattened_frame_keypoints = np.array(frame_keypoints).flatten()
    return flattened_frame_keypoints, result
