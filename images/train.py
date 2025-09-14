from ultralytics import YOLO

# Load YOLOv8n pretrained model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="dataset/data.yaml",  # path to your data.yaml
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu",              # or "0" for GPU if available
    name="pathhole_model"      # folder name for this training run
)
