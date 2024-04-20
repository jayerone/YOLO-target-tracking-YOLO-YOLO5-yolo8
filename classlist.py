import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

# Get class names
class_names = model.names

# Print class names with corresponding indices
print(model.names)

