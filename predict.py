from ultralytics import YOLO
from pathlib import Path
import cv2

# save_path = './output/detect.txt'
# images_path = 'E:\\project\\datasets\\shuibiao\\images\\val'
# images_path = './tests/huaxiangroad/0421.mp4'
# images_path = 'E:\\test\\watermeter\\0001.jpg'
# images_path = 'E:\\test\\1bk.mp4'
images_path = './tests/1bk.mp4'
model = YOLO('./weights/detect/best_pipe.pt')
# model = YOLO("yolov8n.pt")
res = list(model(images_path, save=True, stream=True))

# for r in res:
#     boxes = r.boxes  # Box object used for boundary box output
#     masks = r.masks  # Mask object used to split mask output
#     probs = r.probs  # Category probability for classification output
# results = model.predict(images_path)
# print("******************", results[0])


# for result in res[0]:
#    print('******************', result.boxes.cls, result.boxes.conf, result.boxes.xyxy)

# res_plotted = res[0].plot()
# cv2.namedWindow('result', cv2.WINDOW_NORMAL)
# cv2.imshow("result", res_plotted)
# cv2.waitKey(0)
