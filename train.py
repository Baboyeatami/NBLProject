from ultralytics import YOLO

# Load the YOLOv8 model (choose 'yolov8n', 'yolov8s', 'yolov8m', etc.)
model = YOLO("my_training_runs/exp1/weights/last.pt")  # Using the YOLOv8 Nano model as a base

# Train the model
results = model.train(
    data="data.yaml",  # Path to your dataset YAML file
    epochs=120,                         # Number of training epochs
    batch=4,                          # Batch size
    imgsz=640,                         # Image size (default is 640x640)
    device="cpu",
    project="my_training_runs",  # Custom save directory
    name="exp1",  # Experiment name
    save=True,
    # Use GPU if available
    resume = True  # Use GPU if available

)

# Save the best model weights
best_model_path = model.ckpt_path
print(f"Best model saved at: {best_model_path}")
