from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\raksh\Downloads\best (3).pt")

try:
    with open("config/live_video_src.txt", "r") as file:
        video_source = file.read().strip()
        if not video_source:
            raise ValueError("Video source is empty. Please provide a valid source in the file.")
except FileNotFoundError:
    raise FileNotFoundError("The configuration file 'config/live_video_src.txt' was not found.")
except Exception as e:
    raise Exception(f"Error reading video source: {e}")

if video_source.isdigit():
    video_source = int(video_source)

cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    raise Exception(f"Error: Unable to open video source '{video_source}'. Check the file path or webcam connection.")

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to read frame.")
        break

    results = model.predict(source=frame, show=False)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection Output", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
