from ultralytics import YOLO
import pandas as pd

# Load the trained model (update path if needed)
model = YOLO("runs/detect/pathhole_model2/weights/best.pt")  # or last.pt

# Validate the model
metrics = model.val(data="dataset/data.yaml")

# Print metrics in console
print("Validation Metrics:")
print(metrics)

# Save metrics to CSV
results = {
    "precision": metrics.box.map50,   # Precision at IoU=0.5
    "recall": metrics.box.recall,     # Recall
    "mAP50": metrics.box.map50,       # mAP@0.5
    "mAP50-95": metrics.box.map,      # mAP@0.5:0.95
}
df = pd.DataFrame([results])
df.to_csv("validation_results.csv", index=False)

print("\nMetrics saved to validation_results.csv")
