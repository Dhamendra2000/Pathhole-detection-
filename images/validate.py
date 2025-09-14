from ultralytics import YOLO

# Load best trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Validate and print metrics
metrics = model.val(data="dataset/data.yaml")
print(metrics)  # prints Precision, Recall, mAP
