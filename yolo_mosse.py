import cv2
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Detect objects and initialize trackers
    if len(trackers) == 0:  # You can change this condition to reinitialize trackers less/more often
        results = model(frame)
        for det in results.xyxy[0]:
            if det[4] >= 0.5 and det[-1] == 2:  # Assuming 2 is the class index for 'car'
                x1, y1, x2, y2 = map(int, det[:4])
                tracker = cv2.legacy.TrackerMOSSE_create()
                bbox = (x1, y1, x2-x1, y2-y1)
                tracker.init(frame, bbox)
                trackers.append(tracker)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
