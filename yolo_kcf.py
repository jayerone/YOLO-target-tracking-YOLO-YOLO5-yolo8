import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

# Initialize video
cap = cv2.VideoCapture('/D:\program_file\YoloTracking-main\same_instance.mp4')

# Initialize tracker for each car detected
trackers = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Update trackers if they are already initialized
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Every N frames, detect objects and reinitialize trackers
    if len(trackers) == 0:
        results = model(frame)
        for det in results.xyxy[0]:
            if det[-1] >= 2:  # Assuming 2 is the class index for 'car'
                x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                tracker = cv2.TrackerKCF_create()
                bbox = (x, y, w, h)
                tracker.init(frame, bbox)
                trackers.append(tracker)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
