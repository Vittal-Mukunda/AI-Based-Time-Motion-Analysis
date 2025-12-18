from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    # --- 1. Path to your Roboflow YOLOv8 dataset YAML ---
    data_file_path = r"ADD FILE LOCATION HERE\data.yaml"

    # --- 2. Load a YOLOv8 model to start training ---
    print("Loading 'yolov8s.pt' model...")
    model = YOLO("yolov8s.pt")
    print("Model loaded.")

    # --- 3. Begin training ---
    print("Starting training on Flow Process Chart dataset...")
    results = model.train(
        data=data_file_path,          # path to data.yaml
        epochs=100,                   # number of training epochs
        imgsz=640,                    # image size
        batch=8,                      # batch size
        name="flow_process_chart_v1"  # name for "runs/" folder
    )

    # --- 4. Training complete ---
    print("Training complete!")
    print("Your trained model is located at:")
    print(r"runs\detect\flow_process_chart_v1\weights\best.pt")
