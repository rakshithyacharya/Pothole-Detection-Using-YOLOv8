import cv2
import time
from ultralytics import YOLO

model = YOLO(r"C:\Users\raksh\Downloads\best (3).pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Starting live pothole detection. Press 'q' to quit.")

window_name = 'Live Pothole Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow(window_name, 800, 600)
cv2.moveWindow(window_name, 100, 100) 

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        results = model(frame)
        
        annotated_frame = results[0].plot()

        cv2.putText(annotated_frame, "Live Pothole Detection", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        yield rgb_frame
        cv2.imshow(window_name, annotated_frame)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
for frame in generate_frames():
    pass
cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")