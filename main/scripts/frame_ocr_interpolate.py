import numpy as np
from scipy.interpolate import CubicSpline, interp1d, Akima1DInterpolator
import scipy.signal
import re
import cv2
from paddleocr import PaddleOCR
from scipy.signal import medfilt, savgol_filter

from defect_infoSegmentold import FrameData


class FrameOCR:
    def __init__(self, video_path, ocr_model_dir):
        self.video_path = video_path
        self.ocr_model_dir = ocr_model_dir
        self.ocr = PaddleOCR(rec_model_dir=self.ocr_model_dir, use_angle_cls=True, lang='ch')
        self.frame_data_dict = None

    def process_frame(self, cap, frame_data_dict):
        self.frame_data_dict = frame_data_dict
        frame_id = 0
        distances = {}  # 存储帧ID和对应的距离

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id not in frame_data_dict:
                frame_data_dict[frame_id] = FrameData(frame_id)

            frame_data = frame_data_dict[frame_id]
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
                        distances[frame_id] = current_distance
                        break  # 只处理第一个匹配的距离
                    texts.append(text)

            print("Frame ID:", frame_id)
            print("Distance:", frame_data.distance)
            print("OCR Texts:", texts)

            frame_id += 1

        return distances

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

    def process_video(self, all_frame_data):
        cap = cv2.VideoCapture(self.video_path)
        distances = self.process_frame(cap, all_frame_data)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.interpolate_distances(distances, total_frames)
        all_frame_data = self.frame_data_dict
        cap.release()
        return all_frame_data

