from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')  # build from YAML and transfer weights
    model.train(data='pipe_cctv_text.yaml', epochs=20, imgsz=640)

if __name__ == '__main__':
    main()