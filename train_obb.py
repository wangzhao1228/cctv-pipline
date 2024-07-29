from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-obb.yaml').load('yolov8n-obb.pt')  # build from YAML and transfer weights
    model.train(data='pipe_cctv_obb.yaml', epochs=2000, imgsz=640)

if __name__ == '__main__':
    main()