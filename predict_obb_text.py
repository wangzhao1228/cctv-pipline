import psutil
import GPUtil
from ultralytics import YOLO
from tabulate import tabulate
images_path = './tests/huaxiangroad/4.mp4'
model = YOLO('./weights/obbText/best_obb_text.pt')
# model = YOLO("yolov8n.pt")
# res = model(images_path,save=True,stream=True)
res = model(images_path,save=False,stream=True)

# 使用字典初始化计数器
defect_counts = {0: 0, 1: 0, 2: 0}
class_names = {0: "RoadText", 1: "RobotText", 2: "TimeText"}

for r in res:
    if r.obb is not None:
        for obb in r.obb:
            class_id = int(obb.cls.item())
            if class_id in defect_counts:
                defect_counts[class_id] += 1

# 输出结果
for class_id, count in defect_counts.items():
    print(f"检测到 {class_names[class_id]}: {count}个")
