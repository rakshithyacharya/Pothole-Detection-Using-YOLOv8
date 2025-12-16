import streamlit as st
import pandas as pd 
import geocoder
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from mail import send_email
from pothole_detection import detect_from_video 
from ultralytics import YOLO
import cv2
import time
import requests 
import json 
import sys 

# --- Setup ---
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load Model
# Ensure this path matches your computer's actual path
model = YOLO(r"C:\Users\raksh\Downloads\best (3).pt")

st.set_page_config(page_title="Pothole Detection System", layout="wide")

page = st.sidebar.selectbox("Pages Menu", options=['Home', 'Using Image', 'Using Video', 'Live Camera'])

# --- Helper Functions ---
def get_exif_data(image):
    exif_data = {}
    try:
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]
                    exif_data[decoded] = gps_data
                else:
                    exif_data[decoded] = value
    except: pass
    return exif_data

def get_lat_lon(exif_data):
    lat, lon = None, None
    if "GPSInfo" not in exif_data: return None
    gps_info = exif_data["GPSInfo"]
    def convert_to_degrees(value):
        try:
            d = float(value[0].numerator) / float(value[0].denominator)
            m = float(value[1].numerator) / float(value[1].denominator)
            s = float(value[2].numerator) / float(value[2].denominator)
            return d + (m / 60.0) + (s / 3600.0)
        except: return 0.0
    try:
        gps_latitude = gps_info.get("GPSLatitude")
        gps_latitude_ref = gps_info.get("GPSLatitudeRef")
        gps_longitude = gps_info.get("GPSLongitude")
        gps_longitude_ref = gps_info.get("GPSLongitudeRef")
        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = convert_to_degrees(gps_latitude)
            lon = convert_to_degrees(gps_longitude)
            if gps_latitude_ref != "N": lat = -lat
            if gps_longitude_ref != "E": lon = -lon
            return [lat, lon]
    except: return None
    return None

def get_fallback_location():
    try:
        g = geocoder.ip('me')
        if g.latlng: return g.latlng
    except: pass
    return [0, 0]

def save_uploaded_image(image_file):
    img = Image.open(image_file)
    img_path = "uploads/image.jpg"
    try:
        exif = img.info.get('exif')
        if exif: img.save(img_path, exif=exif)
        else: img.save(img_path)
    except: img.save(img_path)
    exif_data = get_exif_data(img)
    gps_coords = get_lat_lon(exif_data)
    if gps_coords: return img, gps_coords, img_path
    else: return img, get_fallback_location(), img_path

def get_pothole_info(auto_location):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Report Details")
    highway_type = st.sidebar.selectbox("Road Type:", options=["National Highway", "Local Road"])
    size = st.sidebar.selectbox("Pothole Size:", options=["Small", "Medium", "Large"])
    position = st.sidebar.selectbox("Position:", options=["Center", "Left Edge", "Right Edge"])
    return auto_location, highway_type, size, position

def register(location, highway_type, size, position, is_video=False):
    data = {"location": location, "highway_type": highway_type, "size": size, "position": position}
    send_email(data, 'rakshithacharoffl@gmail.com', is_video)
    st.success("✅ Report Sent!")
    print(f"[EMAIL] Sent report for {location}")

# ---------------- IMAGE PAGE ----------------
if page == 'Using Image':
    st.title("Image Analysis")
    choice_upload = st.sidebar.selectbox("Method", options=['Upload Image', 'Open Camera'])

    if choice_upload == 'Upload Image':
        image_file = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'])
        if image_file is not None:
            col1, col2 = st.columns(2)
            img, gps_coords, img_path_local = save_uploaded_image(image_file)
            
            # FIX: changed use_column_width to use_container_width
            col1.image(img, caption="Original", use_container_width=True)

            with st.spinner("Analyzing..."):
                # --- METRICS START ---
                start_time = time.time()
                results = model(img_path_local)
                end_time = time.time()
                
                inference_time_ms = (end_time - start_time) * 1000
                confidence = 0.0
                detections = 0
                if len(results) > 0:
                    boxes = results[0].boxes
                    detections = len(boxes)
                    if detections > 0:
                        confidence = float(boxes.conf[0]) * 100
                # --- METRICS END ---

                # --- TERMINAL PRINT FOR IMAGE ---
                print("\n" + "="*40)
                print(f" IMAGE ANALYSIS RESULTS")
                print(f"="*40)
                print(f" Speed    : {inference_time_ms:.2f} ms")
                print(f" Accuracy : {confidence:.2f} %")
                print(f" Found    : {detections} potholes")
                print("="*40 + "\n")
                # ------------------------------

                res_plotted = results[0].plot()
                cv2.imwrite("results/image_result.jpg", res_plotted)
                
                # FIX: changed use_column_width to use_container_width
                col2.image("results/image_result.jpg", caption="Result", use_container_width=True)
                
                st.write(f"**Speed:** {inference_time_ms:.2f} ms | **Confidence:** {confidence:.2f}%")

            location, highway_type, size, position = get_pothole_info(gps_coords)
            st.map(pd.DataFrame({'lat': [location[0]], 'lon': [location[1]]}))
            if st.sidebar.button("Submit Report"):
                register(location, highway_type, size, position)

    elif choice_upload == 'Open Camera':
        img_file_buffer = st.camera_input("Take Picture")
        if img_file_buffer:
            img = Image.open(img_file_buffer)
            img.save("uploads/image.jpg")
            gps_coords = get_fallback_location()

            start_time = time.time()
            results = model("uploads/image.jpg")
            end_time = time.time()

            inference_time_ms = (end_time - start_time) * 1000
            confidence = 0.0
            if len(results[0].boxes) > 0:
                confidence = float(results[0].boxes.conf[0]) * 100

            # --- TERMINAL PRINT FOR CAMERA SNAPSHOT ---
            print("\n" + "="*40)
            print(f" SNAPSHOT ANALYSIS RESULTS")
            print(f"="*40)
            print(f"Speed    : {inference_time_ms:.2f} ms")
            print(f"Accuracy : {confidence:.2f} %")
            print("="*40 + "\n")

            st.image(results[0].plot(), caption="Result")
            location, highway_type, size, position = get_pothole_info(gps_coords)
            if st.sidebar.button("Submit Report"): register(location, highway_type, size, position)

# ---------------- VIDEO PAGE ----------------
elif page == 'Using Video':
    st.title("Video Analysis")
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if video_file:
        with open("uploads/video.mp4", "wb") as f: f.write(video_file.read())
        if st.button("Process Video"):
            # This function in pothole_detection.py will now handle the printing
            output_path = detect_from_video("uploads/video.mp4", model)
            if output_path: st.video(output_path)

# ---------------- LIVE PAGE ----------------
elif page == 'Live Camera':
    st.title("Live Detection")
    if 'camera_running' not in st.session_state: st.session_state.camera_running = False
    
    col1, col2 = st.columns(2)
    if col1.button("Start"): st.session_state.camera_running = True
    if col2.button("Stop"): st.session_state.camera_running = False

    if st.session_state.camera_running:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        prev_time = 0
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret: break
            
            results = model(frame, conf=0.4, verbose=False)
            annotated_frame = results[0].plot()
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            conf_val = 0.0
            if len(results[0].boxes) > 0:
                conf_val = float(results[0].boxes.conf[0]) * 100

            # --- LIVE TERMINAL UPDATE ---
            sys.stdout.write(f"\r LIVE > FPS: {fps:.1f} | Conf: {conf_val:.1f}%   ")
            sys.stdout.flush()

            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # FIX: changed use_column_width to use_container_width
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        cap.release()
else:
    st.title('Pothole Detection')
    st.markdown("> Select any choice from the sidebar to proceed")
    st.image("image1.jpg")
    st.markdown(""" ## Detecting Potholes on Road using YOLOv8 Model
    Features:
    - Detects Potholes From Images
    - Detects Potholes Using Live Camera Feed
    - Detects Potholes From Uploaded Videos
    - Reports Pothole Data through email
    - Automatically Retrieves Location Information via IP Address           
    """)