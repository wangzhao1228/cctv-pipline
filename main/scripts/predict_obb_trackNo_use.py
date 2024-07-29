import cv2
from ultralytics import YOLO
from defect_info import FrameData

class VideoTracker:
    def __init__(self, model_path, video_path, output_path):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.output_path = output_path
        self.frame_id = 0

        # Video output setup
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    def track_frames(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_data = FrameData(self.frame_id)
            results = self.model.track(frame, persist=True)

            for result in results:
                class_id = result['class']
                defect_types = {0: '气囊', 1: '异物', 2: '管瘤', 3: '腐蚀', 4: '沉积', 5:'漏点',6:'连接'}
                defect_type = defect_types.get(class_id, "未知")

                # 假设每个检测到的对象都计数为1
                frame_data.add_defect(defect_type)

            annotated_frame = results.plot()  # 可视化结果
            self.out.write(annotated_frame)
            cv2.imshow("Tracking", annotated_frame)
            self.frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

# 示例使用
if __name__ == "__main__":
    model_path = '/path/to/model.pt'
    video_path = '/path/to/video.mp4'
    output_path = '/path/to/output.mp4'
    tracker = VideoTracker(model_path, video_path, output_path)
    tracker.track_frames()
