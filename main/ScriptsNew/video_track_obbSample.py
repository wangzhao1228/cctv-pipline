import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
from flask import logging
from line_profiler import profile
from scipy.interpolate import interp1d, Akima1DInterpolator
from scipy.signal import savgol_filter, medfilt
from skimage.metrics import structural_similarity as ssim

from ultralytics import YOLO
from defect_infoSegment0911 import FrameData
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor
import matplotlib.pyplot as plt
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

    def is_key_frame(self, frame1, frame2, threshold=1):
        """
        通过比较两张图片的相似度来判断是否是关键帧。

        :param frame1: 第一张图片（上一帧）
        :param frame2: 第二张图片（当前帧）
        :param threshold: SSIM 相似度阈值，低于该值时认为是关键帧
        :return: 是否是关键帧
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 假设gray1和gray2是8位图像，即像素值范围为0到255
        score, _ = ssim(gray1, gray2, data_range=gray1.max() - gray1.min(), full=True)
        return score < threshold

    def track_video(self, sampleFrequency=1):
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))
        frame_id = 0  # 用于跟踪从视频读取的每一帧
        process_id = 0  # 用于跟踪实际处理的帧编号
        total_processed_frames = 0  # 实际处理的帧数
        last_distance = 0

        last_frame = None

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_id % sampleFrequency == 0:  # 每隔一帧处理一次
                if last_frame is None or self.is_key_frame(last_frame, frame):
                    # Yolo track
                    last_frame = frame
                    total_processed_frames += 1  # 增加已处理帧数

                    if process_id not in self.frame_data_dict:
                        self.frame_data_dict[process_id] = FrameData(process_id)
                    frame_data = self.frame_data_dict[process_id]

                    results = self.model.track(frame, persist=True, conf=0)
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
                                conf_threshold = self.conf_thresholds.get(defect_name, 0)
                                if conf >= conf_threshold:
                                    frame_data.add_defect(defect_name, defect_id, width, height, x_center, y_center,
                                                          rot, conf)

                        if hasattr(result, 'plot'):
                            annotated_frame = result.plot()  # 对每个结果调用plot
                            out.write(annotated_frame)  # 将标注后的帧写入输出文件

                    # 更新用于跟踪实际处理的帧编号
                    process_id += 1

                    # ocr
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ocr_results = self.ocr.ocr(processed_frame, cls=True)

                    distance_found = False
                    for line in ocr_results:
                        if line is None:
                            continue
                        for element in line:
                            text = element[1][0]
                            match = re.search(r"距离\s*[:：]\s*([-+]?\d+\.\d+)\s*M", text)
                            if match:
                                current_distance = float(match.group(1))
                                frame_data.set_distance(current_distance)
                                self.distances[frame_id] = current_distance
                                last_distance = current_distance
                                distance_found = True
                                break  # 只处理第一个匹配的距离
                        if distance_found:
                            break
                    # 如果未找到距离信息，使用上一帧的距离值
                    if not distance_found:
                        frame_data.set_distance(last_distance)
                        self.distances[frame_id] = last_distance

                frame_id += 1  # 总帧计数器
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.interpolate_distances(self.distances, total_processed_frames)
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

    def filter_anomalies_with_segmented_interpolation(self, values, segment_size=1000, mad_threshold=10):
        filtered_values = np.copy(values)

        def mad_based_outlier(points, threshold=mad_threshold):
            if len(points) < 3:
                return np.array([False] * len(points))
            median = np.median(points)
            diff = np.abs(points - median)
            mad = np.median(diff)
            if mad == 0:
                return np.array([False] * len(points))
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
                if segment_anomaly_indices.size > 0:
                    filtered_values[start:end][anomalies] = linear_interpolator(segment_anomaly_indices)
            else:
                # 如果全部都是异常值，就使用整个数据段的中位数
                filtered_values[start:end] = np.median(segment_values)

        return filtered_values

    def interpolate_distances(self, distances, total_frames, json_output_path='processed_distances.json'):
        valid_indices = np.array(list(distances.keys()))
        valid_values = np.array(list(distances.values()))

        # 将数据按照帧号排序
        sorted_order = np.argsort(valid_indices)
        valid_indices = valid_indices[sorted_order]
        valid_values = valid_values[sorted_order]

        # 应用自定义的异常值滤除函数
        filtered_values = self.filter_anomalies_with_segmented_interpolation(valid_values)

        # 数据预处理：使用中位数滤波
        filtered_values = medfilt(filtered_values, kernel_size=5)

        # 插值选择：使用Akima插值或分段线性插值避免过度振荡
        if len(valid_indices) > 1:
            interpolator = Akima1DInterpolator(valid_indices, filtered_values)
            interpolated_distances = interpolator(np.arange(total_frames))
        else:
            interpolated_distances = np.full(total_frames, np.median(filtered_values))

        # 填充前面的NaN值
        if np.isnan(interpolated_distances[0]):
            first_valid_index = np.where(~np.isnan(interpolated_distances))[0][0]
            interpolated_distances[:first_valid_index] = interpolated_distances[first_valid_index]

        # 填充后面的NaN值
        if np.isnan(interpolated_distances[-1]):
            last_valid_index = np.where(~np.isnan(interpolated_distances))[0][-1]
            interpolated_distances[last_valid_index + 1:] = interpolated_distances[last_valid_index]

        # 检测和修正局部异常值（变化超过1米）
        for i in range(1, len(interpolated_distances) - 1):
            if np.abs(interpolated_distances[i] - interpolated_distances[i - 1]) > 1 and \
                    np.abs(interpolated_distances[i] - interpolated_distances[i + 1]) > 1:
                interpolated_distances[i] = (interpolated_distances[i - 1] + interpolated_distances[i + 1]) / 2

        # 迭代修正，确保所有两帧之间的变化不超过0.5米
        max_distance_change_per_frame = 0.5
        changes_exceeding = True
        iteration = 0
        max_iterations = 100  # 防止无限循环
        iteration_exceeded = False

        # 添加记录修改的列表
        modifications = []

        while changes_exceeding and iteration < max_iterations:
            changes_exceeding = False
            for i in range(1, total_frames):
                change = abs(interpolated_distances[i] - interpolated_distances[i - 1])
                if change > max_distance_change_per_frame:
                    changes_exceeding = True
                    original_value = interpolated_distances[i]
                    if i < total_frames - 1:
                        interpolated_distances[i] = (interpolated_distances[i - 1] + interpolated_distances[i + 1]) / 2
                    else:
                        # 如果是最后一帧，使用前一帧的值
                        interpolated_distances[i] = interpolated_distances[i - 1]
                    # 记录修改信息
                    modifications.append({
                        "frame": i,
                        "original_value": original_value,
                        "corrected_value": interpolated_distances[i]
                    })
            iteration += 1

        if iteration == max_iterations and changes_exceeding:
            iteration_exceeded = True
            print("达到最大迭代次数，可能存在无法完全修正的异常值。")

        # 使用更大的平滑窗口，仅对插值后的数据进行平滑
        window_size = 15  # 增大窗口大小，例如15帧
        interpolated_distances = pd.Series(interpolated_distances).rolling(
            window=window_size, min_periods=1, center=True).mean().values

        # 更新 frame_data_dict 中的距离值
        for i in range(total_frames):
            if i in self.frame_data_dict:
                self.frame_data_dict[i].set_distance(interpolated_distances[i])
            else:
                # 如果帧数据不存在，创建新的 FrameData 对象
                self.frame_data_dict[i] = FrameData(i)
                self.frame_data_dict[i].set_distance(interpolated_distances[i])

        # 绘制距离值随帧序号的变化图，标记异常帧
        self.plot_distances(valid_indices, valid_values, interpolated_distances, modifications)

        # 将处理后的距离和迭代信息保存到JSON
        output_data = {
            "distances": interpolated_distances.tolist(),
            "iteration_exceeded": iteration_exceeded,
            "modifications": modifications  # 添加修改记录
        }

        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"处理后的距离数据和迭代信息已保存到 {json_output_path}")

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

    def get_all_frame_data(self):
        return self.frame_data_dict
