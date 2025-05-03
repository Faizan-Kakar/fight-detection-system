#  1. 🔍 Fight Detection System
An AI-powered surveillance system that detects and alerts incidents of physical violence in real-time using computer vision and deep learning.

## 2. Table of Contents
- [Overview](#3-overview)
- [Tech Stack](#4-tech-stack)
- [Project Structure](#5-project-structure)
- [How It Works](#6-how-it-works)
- [Installation](#7-installation)
- [Usage](#8-usage)
- [Future Enhancements](#9-future-enhancements)
- [Author](#10-author)




## 3. 🧠 Overview
The Violence Detection System is an AI-powered surveillance solution designed to automatically identify physical violence in real-time using video feeds. It extracts frames from live or recorded videos, processes them using pose estimation to detect human keypoints, and classifies the activity through a deep learning model. When violent behavior is detected, the system triggers visual alerts and logs the incidents with timestamps. A user-friendly dashboard built with Streamlit displays incident summaries and analytical insights such as total incident count and model accuracy. This project aims to enhance public safety in areas like schools, malls, and streets through automated monitoring.s a real-time intelligent surveillance system that detects incidents from camera feeds using pose estimation and alert mechanisms. It enables automated monitoring and quick response in sensitive environments such as campuses, offices, or public spaces.



## 4. ⚙️ Tech Stack
- Python
- OpenCV
- MediaPipe / Pose Estimation
- NumPy, Matplotlib
- Streamlit


## 5. 🗂️ Project Structure
```text
Final Year Project/
│
├── components/                     # 📦 UI Components for Streamlit app
│   ├── alert_box.py               # 🔔 Displays alert messages (e.g., incident detected)
│   ├── analytics_panel.py         # 📊 Shows analytics like total incidents, accuracy, etc.
│   └── incident_cards.py          # 🧾 Displays incident summaries in card format
│
├── data/                          # 📂 Placeholder for datasets (videos, frames, etc.)
│
├── models/                        # 🧠 Pretrained or custom-trained ML/DL models
│
├── src/                           # 🛠️ Core logic and model inference scripts
│   └── predictions.py             # 🔍 Handles prediction logic using the trained models
│
├── utills/                        # 🧰 Utility scripts for data and session handling
│   ├── data_preprocessing.py      # 🧼 Cleans and prepares data for training/prediction
│   ├── frame_to_keypoints.py      # 🎯 Converts video frames to keypoints for analysis
│   └── session_utill.py           # 💾 Manages Streamlit session state
│
├── app.py                         # 🚀 Main entry point for running the Streamlit web app
│
├── requirements.txt               # 📜 Python dependencies required to run the project
```

## 5. 🧪 How It Works
1. Extract frames from live feed or video.
2. Convert each frame to keypoints using pose estimation.
3. Preprocess and feed keypoints to trained model.
4. Classify actions and detect abnormal incidents.
5. Display alerts or analytics on UI.

## 7. 🚀 Installation
Follow the steps below to set up the project on your local machine:
1. **Clone the Repository**
```bash
git clone https://github.com/Faizan-Kakar/violence-detection-system.git
cd violence-detection-system
```
2. **Open the project in your preferred IDE (e.g., Visual Studio Code)**
- Launch VS code
- Go to File → Open Folder... → Select the cloned project folder.

3. **Open a terminal inside the IDE**
- In VS Code, go to Terminal → New Terminal.

4. **Install all required dependencies**
```bash
pip install -r requirements.txt
```

## 8. 🧭 Usage
Run the main file
```bash
streamlit run app.py
```


---

<!-- ### 9. 📸 Sample Output -->
<!-- Add a sample screenshot or console output.-->
<!-- ```markdown -->
<!-- ## 📸 Sample Output -->
<!-- ![Preview](path_to_sample_image_or_gif.gif) -->

## 9. 📌 Future Enhancements
- Train on more diverse datasets
- Add real-time database or alert system

## 👤 Author
**Faizan Khan**  
AI/ML Developer | Final Year Student  
[LinkedIn]( www.linkedin.com/in/faizan-kakar ) • [Portfolio](https://github.com/Faizan-Kakar)

