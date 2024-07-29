import psutil
from ultralytics import YOLO
images_path = './tests/huaxiangroad/FM.mp4'
model = YOLO('./weights/obb/best.pt')
# model = YOLO("yolov8n.pt")
# res = model(images_path,save=True,stream=True)
res = model(images_path,save=False,stream=True)

# print("******************", res[0].obb)

# 使用字典初始化计数器
defect_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
class_names = {0: "QN", 1: "YW", 2: "GL", 3: "FS", 4: "CJ"}

for r in res:
    if r.obb is not None:
        for obb in r.obb:
            class_id = int(obb.cls.item())
            if class_id in defect_counts:
                defect_counts[class_id] += 1

# 输出结果
for class_id, count in defect_counts.items():
    print(f"检测到 {class_names[class_id]}: {count}个")


# 在处理前后打印CPU和内存使用情况
def print_system_usage():
    print(f"CPU使用率: {psutil.cpu_percent()}%")
    print(f"内存使用率: {psutil.virtual_memory().percent}%")

print("处理前的系统资源使用情况:")
print_system_usage()