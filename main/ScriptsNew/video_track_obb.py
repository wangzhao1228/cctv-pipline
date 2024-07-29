from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
from line_profiler import profile

from ultralytics import YOLO
from defect_infoSegment import FrameData
import numpy as np
from scipy.interpolate import CubicSpline, interp1d, Akima1DInterpolator
import scipy.signal
import re
from paddleocr import PaddleOCR


class VideoTracker:
    def __init__(self, video_path, model_path, ocr_model_path,output_path, conf_thresholds, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.video_path = video_path
        self.model = YOLO(model_path).to(device)
        self.output_path = output_path
        self.ocr_model_dir = ocr_model_path
        self.conf_thresholds = conf_thresholds
        self.ocr = PaddleOCR(rec_model_dir=self.ocr_model_dir, use_angle_cls=True, use_gpu = True, lang='ch')
        self.frame_data_dict = {}  # 使用字典存储帧数据
        self.distances = {}  # 使用字典存储每帧距离数据

    def track_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))
        frame_id = 0


        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            # Yolo track
            if frame_id not in self.frame_data_dict:
                self.frame_data_dict[frame_id] = FrameData(frame_id)
            frame_data = self.frame_data_dict[frame_id]

            results = self.model.track(frame, persist=True, conf=0.2)
            for result in results:  # 处理每个结果对象
                if hasattr(result, 'obb') and result.obb and result.obb.is_track:
                    class_ids = result.obb.cls.numpy()
                    track_ids = result.obb.id.numpy() if result.obb.is_track else None
                    confs = result.obb.conf.numpy()
                    for idx, class_id in enumerate(class_ids):
                        defect_id = track_ids[idx] if track_ids is not None else None
                        defect_name = self.get_defect_name(class_id)
                        x_center = result.obb.xywhr[idx, 0].item()
                        y_center = result.obb.xywhr[idx, 1].item()
                        width = result.obb.xywhr[idx, 2].item()
                        height = result.obb.xywhr[idx, 3].item()
                        rot = result.obb.xywhr[idx, 4].item()
                        conf = confs[idx].item()
                        conf_threshold = self.conf_thresholds.get(defect_name, 0.35)
                        if conf >= conf_threshold:
                            frame_data.add_defect(defect_name, defect_id, width, height, x_center, y_center, rot, conf)

                if hasattr(result, 'plot'):
                    annotated_frame = result.plot()  # 对每个结果调用plot
                    out.write(annotated_frame)  # 将标注后的帧写入输出文件
                    # cv2.imshow("YOLOv8 Tracking", annotated_frame)
            # Yolo end
            # ocr
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ocr_results = self.ocr.ocr(processed_frame, cls=True)
            texts = []

            for line in ocr_results:
                for element in line:
                    text = element[1][0]
                    match = re.search(r"距离\s*[:：]\s*([-+]?\d+\.\d+)\s*M", text)
                    if match:
                        current_distance = float(match.group(1))
                        frame_data.set_distance(current_distance)
                        self.distances[frame_id] = current_distance
                        break  # 只处理第一个匹配的距离
                    texts.append(text)

            # ocr end
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_id += 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.interpolate_distances(self.distances, total_frames)
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def get_defect_name(self, class_id):
        defect_mapping = {
            0: 'QN',  # 气囊
            1: 'YW',  # 异物
            2: 'GL',  # 管瘤
            3: 'FS',  # 腐蚀
            4: 'CJ',  # 沉积
            5: 'ZJ',  # 直径
            6: 'ZG',  # 支管
            7: 'LGL',  # 重度管瘤
            8: 'MGL',  # 中度管瘤
            9: 'SGL',  # 轻度管瘤
            10: 'LFS',  # 重度腐蚀
            11: 'MFS',  # 中度腐蚀
            12: 'SFS',  # 轻度腐蚀
            13: 'SZ',  # 绳子
            14: 'FM',  # 阀门
            15: 'TL',  # 脱落管壁
            16: 'WZ',  # 弯折管道
            17: 'PS',  # 破损管壁
            18: 'ZC',  # 正常

        }
        return defect_mapping.get(class_id, '未知')

    def filter_anomalies_with_segmented_interpolation(self, values, segment_size=1000):
        filtered_values = np.copy(values)

        def mad_based_outlier(points, threshold=3.5):
            if len(points) < 3:
                return np.array([False] * len(points))
            median = np.median(points)
            diff = np.abs(points - median)
            mad = np.median(diff)
            modified_z_score = 0.6745 * diff / mad
            return modified_z_score > threshold

        # 处理每个分段
        for start in range(0, len(values), segment_size):
            end = min(start + segment_size, len(values))
            segment_values = values[start:end]

            # 使用MAD检测异常值
            anomalies = mad_based_outlier(segment_values)

            # 正常值的索引
            normal_indices = np.where(~anomalies)[0]
            normal_values = segment_values[normal_indices]

            if normal_indices.size > 0:
                # 创建一个线性插值模型
                linear_interpolator = interp1d(normal_indices, normal_values,
                                               kind='linear', fill_value='extrapolate', bounds_error=False)
                # 使用插值模型填充异常值
                segment_anomaly_indices = np.where(anomalies)[0]
                filtered_values[start:end][anomalies] = linear_interpolator(segment_anomaly_indices)
            else:
                # 如果全部都是异常值，就使用整个数据段的中位数
                filtered_values[start:end] = np.median(segment_values)

        return filtered_values

    def interpolate_distances(self, distances, total_frames):
        valid_indices = np.array(list(distances.keys()))
        valid_values = np.array(list(distances.values()))

        # 应用自定义的异常值滤除函数
        valid_values = self.filter_anomalies_with_segmented_interpolation(valid_values)

        # 数据预处理：使用中位数滤波
        filtered_values = scipy.signal.medfilt(valid_values, kernel_size=5)

        # 插值选择：使用Akima插值或分段线性插值避免过度振荡
        if len(valid_indices) > 1:
            interpolator = Akima1DInterpolator(valid_indices, filtered_values)
            interpolated_distances = interpolator(np.arange(total_frames))
        # else:
        #     interpolated_distances = np.full(total_frames, np.median(filtered_values))

        # 填充前面的NaN值
        if np.isnan(interpolated_distances[0]):
            first_valid_index = np.where(~np.isnan(interpolated_distances))[0][0]
            interpolated_distances[:first_valid_index] = interpolated_distances[first_valid_index]

        # 填充后面的NaN值
        if np.isnan(interpolated_distances[-1]):
            last_valid_index = np.where(~np.isnan(interpolated_distances))[0][-1]
            interpolated_distances[last_valid_index + 1:] = interpolated_distances[last_valid_index]

        # 检测和修正异常值
        for i in range(1, len(interpolated_distances) - 1):
            if np.abs(interpolated_distances[i] - interpolated_distances[i - 1]) > 1 and \
                    np.abs(interpolated_distances[i] - interpolated_distances[i + 1]) > 1:
                interpolated_distances[i] = (interpolated_distances[i - 1] + interpolated_distances[i + 1]) / 2

        # 更新距离
        for i in range(total_frames):
            self.frame_data_dict[i].set_distance(interpolated_distances[i])

    def get_all_frame_data(self):
        return self.frame_data_dict
