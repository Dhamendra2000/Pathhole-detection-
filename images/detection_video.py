from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/pathhole_model2/weights/best.pt")

# Open video file
video_path = "detectionvideo.mp4"   # <-- change this to your video path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection on the current frame
    results = model.predict(frame, conf=0.5, verbose=False)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Show the frame in real-time
    cv2.imshow("Pothole Detection - Video", annotated_frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
