import cv2
import re
from paddleocr import PaddleOCR

from defect_infoSegmentold import FrameData


class FrameOCR:
    def __init__(self, video_path, ocr_model_dir):
        self.video_path = video_path
        self.ocr_model_dir = ocr_model_dir
        self.ocr = PaddleOCR(rec_model_dir=self.ocr_model_dir, use_angle_cls=True, lang='ch')

    def process_frame(self, cap, frame_data_dict):
        """处理单个视频帧并更新现有的frame_data_dict"""
        frame_id = 0
        previous_distance = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id not in frame_data_dict:
                frame_data_dict[frame_id] = FrameData(frame_id)

            frame_data = frame_data_dict[frame_id]
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ocr_results = self.ocr.ocr(processed_frame, cls=True)
            distance_extracted = False
            texts = []

            for line in ocr_results:
                for element in line:
                    text = element[1][0]
                    match = re.search(r"距离:(\d+\.\d+)M", text)
                    if match:
                        current_distance = float(match.group(1))
                        if previous_distance is not None and abs(current_distance - previous_distance) > 1:
                            current_distance = previous_distance
                        frame_data.set_distance(current_distance)
                        previous_distance = current_distance
                        distance_extracted = True
                        break  # 只处理第一个匹配的距离
                    texts.append(text)

            print("Frame ID:", frame_data.frame_id)
            print("Distance:", frame_data.distance)
            print("OCR Texts:", texts)

            frame_id += 1  # 更新帧序号

    def process_distances(self, frame_data_dict):
        frame_ids = sorted(frame_data_dict.keys())  # 获取所有帧ID并排序

        # 首先向前填充，保证每帧都有距离数据
        last_known_distance = None
        for frame_id in frame_ids:
            frame_data = frame_data_dict[frame_id]
            if frame_data.distance is not None:
                last_known_distance = frame_data.distance
            elif last_known_distance is not None:
                frame_data.set_distance(last_known_distance)

        # 再向后填充，处理可能在开始处漏填的情况
        next_known_distance = None
        for frame_id in reversed(frame_ids):
            frame_data = frame_data_dict[frame_id]
            if frame_data.distance is not None:
                next_known_distance = frame_data.distance
            elif next_known_distance is not None:
                frame_data.set_distance(next_known_distance)

        # 处理异常距离值：如果距离与前后帧相差较大，则取前后帧的平均值
        for i in range(1, len(frame_ids) - 1):
            prev_frame = frame_data_dict[frame_ids[i - 1]]
            current_frame = frame_data_dict[frame_ids[i]]
            next_frame = frame_data_dict[frame_ids[i + 1]]
            if current_frame.distance is not None:
                if abs(current_frame.distance - prev_frame.distance) > 0.5 and abs(
                        current_frame.distance - next_frame.distance) > 0.5:
                    current_frame.set_distance((prev_frame.distance + next_frame.distance) / 2)

    def process_video(self, all_frame_data):
        """在已有的帧数据基础上，通过OCR添加或更新信息"""
        cap = cv2.VideoCapture(self.video_path)
        self.process_frame(cap, all_frame_data)
        self.process_distances(all_frame_data)  # 对数据进行最终处理s

        cap.release()
