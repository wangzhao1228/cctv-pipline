import copy

import ruptures as rpt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline, interp1d, Akima1DInterpolator
import scipy.signal
import re
import cv2
from paddleocr import PaddleOCR
from scipy.signal import medfilt, savgol_filter

from defect_infoSegment0923 import FrameData


class FrameOCR:
    def __init__(self, video_path, ocr_model_dir):
        self.video_path = video_path
        self.ocr_model_dir = ocr_model_dir
        self.ocr = PaddleOCR(rec_model_dir=self.ocr_model_dir, use_angle_cls=True, lang='ch')
        self.frame_data_dict = None
        self.frame_data_dict_before = None

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
                if line is None:
                    frame_data.set_distance(-10000)
                    texts.append("")
                    continue
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

        self.frame_data_dict_before = frame_data_dict.copy()
        return distances

    def combined_anomaly_detection(self, segment_values, threshold):
        def mad_based_outlier(points, threshold):
            if len(points) < 3:
                return np.array([False] * len(points))
            median = np.median(points)
            diff = np.abs(points - median)
            mad = np.median(diff)
            if mad == 0:
                return np.array([False] * len(points))
            return diff > threshold

        def z_score_based_outlier(points, threshold):
            mean = np.mean(points)
            std_dev = np.std(points)
            if std_dev == 0:
                return np.array([False] * len(points))
            z_scores = np.abs((points - mean) / std_dev)
            return z_scores > threshold

        def iqr_based_outlier(points, threshold):
            Q1 = np.percentile(points, 25)
            Q3 = np.percentile(points, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (points < lower_bound) | (points > upper_bound)

        mad_anomalies = mad_based_outlier(segment_values, threshold)
        z_score_anomalies = z_score_based_outlier(segment_values, threshold)
        iqr_anomalies = iqr_based_outlier(segment_values, threshold)

        # 合并所有检测结果
        combined_anomalies = mad_anomalies | z_score_anomalies | iqr_anomalies
        return combined_anomalies

    def filter_anomalies_with_segmented(self, values, segment_params):
        filtered_values = np.copy(values)

        for segment_size, mad_threshold in segment_params:
            print(f"使用 segment_size={segment_size}, mad_threshold={mad_threshold} 进行异常值检测")
            temp_filtered_values = np.copy(filtered_values)

            for start in range(0, len(values), segment_size):
                end = min(start + segment_size, len(values))
                segment_values = temp_filtered_values[start:end]

                # 提取有效的（非 NaN）数据
                valid_mask = ~np.isnan(segment_values)
                valid_indices = np.where(valid_mask)[0]
                valid_values = segment_values[valid_mask]

                if len(valid_values) >= 3:
                    # 使用异常值检测函数
                    anomalies = self.combined_anomaly_detection(valid_values, mad_threshold)

                    # 标记异常值为 np.nan
                    anomaly_indices_within_valid = np.where(anomalies)[0]
                    anomaly_indices = valid_indices[anomaly_indices_within_valid]
                    temp_filtered_values[start:end][anomaly_indices] = np.nan

            filtered_values = np.copy(temp_filtered_values)

        return filtered_values

    def detect_and_correct_shift(self, indices, values, threshold=30):
        """
        检测并校正数据中的偏移段。

        参数：
        - indices: 帧序列的索引数组。
        - values: 对应的距离值数组。
        - threshold: 判断偏差是否显著的阈值。

        返回：
        - corrected_values: 校正后的距离值数组。
        """
        # 将数据中的 NaN 值处理掉，便于变化点检测
        finite_mask = np.isfinite(values)
        finite_indices = indices[finite_mask]
        finite_values = values[finite_mask]

        # 使用 Ruptures 进行变化点检测
        # 将 model 从 "rbf" 改为 "l2" 或 "normal"
        algo = rpt.Pelt(model="l2").fit(finite_values.reshape(-1, 1))

        # 设置惩罚项，根据数据特性调整
        penalty = 10  # 可以根据需要调整

        result = algo.predict(pen=penalty)

        # 根据检测到的变化点，对每个段落进行偏差校正
        corrected_values = np.copy(values)
        segments = zip([0] + result[:-1], result)
        previous_median = None

        for start_idx, end_idx in segments:
            segment_indices = finite_indices[start_idx:end_idx]
            segment_values = finite_values[start_idx:end_idx]
            segment_median = np.median(segment_values)

            if previous_median is not None:
                # 计算偏差
                shift = segment_median - previous_median

                # 如果偏差超过阈值，认为存在整体偏移，进行校正
                if abs(shift) >= threshold:
                    print(f"在区间 {segment_indices[0]} 到 {segment_indices[-1]} 检测到偏差 {shift}，进行校正。")
                    # 校正该段数据
                    correction_mask = (indices >= segment_indices[0]) & (indices <= segment_indices[-1])
                    corrected_values[correction_mask] -= shift
            else:
                # 第一段，不需要校正
                pass

            previous_median = np.median(corrected_values[finite_mask][start_idx:end_idx])

        return corrected_values

    def interpolate_distances(self, distances, total_frames, json_output_path='processed_distances.json'):
        valid_indices = np.array(list(distances.keys()))
        valid_values = np.array(list(distances.values()))

        # 将数据按照帧号排序
        sorted_order = np.argsort(valid_indices)
        valid_indices = valid_indices[sorted_order]
        valid_values = valid_values[sorted_order]

        # 定义参数组合列表
        segment_params = [
            (40000, 30),
            (10000, 20),
            (5000, 15),
            (1000, 10),
            (500, 5),
            (100, 3),
            (50, 1),
        ]

        # 应用自定义的异常值检测函数，标记异常值为 np.nan
        filtered_values = self.filter_anomalies_with_segmented(valid_values, segment_params)

        # 绘制异常值检测后的有效值
        # plot_filtered_values(valid_indices, filtered_values, title='异常值检测后的有效值')

        # 检查并处理剩余的 NaN 值
        if np.isnan(filtered_values).any():
            finite_indices = np.where(~np.isnan(filtered_values))[0]
            finite_values = filtered_values[finite_indices]

            if len(finite_indices) > 1:
                # 统一使用线性插值填充 NaN 值
                interpolator = interp1d(finite_indices, finite_values, kind='linear', fill_value='extrapolate',
                                        bounds_error=False)
                nan_indices = np.where(np.isnan(filtered_values))[0]
                filtered_values[nan_indices] = interpolator(nan_indices)
            elif len(finite_indices) == 1:
                # 只有一个有效值，用该值填充
                filtered_values[np.isnan(filtered_values)] = finite_values[0]
            else:
                # 没有有效值，用全局均值填充
                overall_mean = np.nanmean(distances)
                filtered_values[:] = overall_mean

            # 检测并校正偏移段
        # corrected_values = self.detect_and_correct_shift(valid_indices, filtered_values)

        # 进行平滑处理
        data_length = len(filtered_values)
        window_length = 11 if data_length >= 11 else max(3, data_length // 2 * 2 + 1)
        polyorder = min(2, window_length - 1)

        # 使用 Savitzky-Golay 滤波器进行平滑
        corrected_values = savgol_filter(filtered_values, window_length=window_length, polyorder=polyorder)

        # 数据预处理：使用中位数滤波
        corrected_values = medfilt(corrected_values, kernel_size=5)

        # 统一对数据进行插值，得到完整的距离序列
        interpolated_frames = np.arange(total_frames)
        interpolator = interp1d(valid_indices, corrected_values, kind='linear', fill_value='extrapolate',
                                bounds_error=False)
        interpolated_distances = interpolator(interpolated_frames)
        # 对插值后的数据进行 NaN 处理
        interpolated_distances = np.nan_to_num(interpolated_distances, nan=np.nanmean(interpolated_distances))

        # 将处理后的距离更新回 self.frame_data_dict
        for idx, frame_id in enumerate(valid_indices):
            if frame_id in self.frame_data_dict:
                self.frame_data_dict[frame_id].set_distance(corrected_values[idx])

        # 将处理后的距离数据返回
        return interpolated_distances

    def plot_distances(self, valid_indices, valid_values, predicted_distances, anomalies=None):
        # 使用 matplotlib 绘制距离随帧号变化的图，标记异常帧
        plt.figure(figsize=(15, 5))
        plt.plot(valid_indices, valid_values, 'o', label='Valid Values')
        plt.plot(np.arange(len(predicted_distances)), predicted_distances, '-', label='Interpolated Distances')
        if anomalies is not None:
            plt.scatter(anomalies, predicted_distances[anomalies], color='red', label='Corrected Anomalies')
        plt.xlabel('Frame ID')
        plt.ylabel('Distance (m)')
        plt.legend()
        plt.title('Distance Values Over Frames with Anomalies')
        plt.show()

    # def process_video(self, all_frame_data):
    #     cap = cv2.VideoCapture(self.video_path)
    #     distances = self.process_frame(cap, all_frame_data)
    #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     self.interpolate_distances(distances, total_frames)
    #     all_frame_data = self.frame_data_dict
    #     cap.release()
    #     return all_frame_data

    def process_video(self, all_frame_data):
        cap = cv2.VideoCapture(self.video_path)

        # 处理帧，提取距离
        distances = self.process_frame(cap, all_frame_data)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 返回处理前的 all_frame_data
        pre_interpolation_data = copy.deepcopy(self.frame_data_dict_before)

        # 插值处理距离
        interpolated_distances = self.interpolate_distances(distances, total_frames)

        # 更新 self.frame_data_dict 中的距离数据
        for frame_id, distance in enumerate(interpolated_distances):
            if frame_id in self.frame_data_dict:
                self.frame_data_dict[frame_id].set_distance(distance)

        # 返回处理后的 all_frame_data
        post_interpolation_data = copy.deepcopy(self.frame_data_dict)

        cap.release()

        return pre_interpolation_data, post_interpolation_data
