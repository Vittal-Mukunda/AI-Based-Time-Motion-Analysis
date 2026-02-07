from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s.pt') 

    results = model.train(
        data=r"C:\Users\vitta\Downloads\flow_process_chart.v2i.yolov8\data.yaml", 
        epochs=50,
        imgsz=640,
        batch=16,       # Increased for GPU
        device=0,       # Changed from 'cpu' to 0
        project='My_Custom_Training',
        name='board_detector_gpu'
    )