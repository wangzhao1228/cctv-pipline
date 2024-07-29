from ultralytics import YOLO


images_path = './tests/huaxiangroad/7.mp4'
model = YOLO('./weights/seg/best.pt')
# model = YOLO("yolov8n.pt")
res = model(images_path,save=True,stream=True)
for r in res:
        boxes = r.boxes  # Box object used for boundary box output
        masks = r.masks  # Mask object used to split mask output
        probs = r.probs  # Category probability for classification output