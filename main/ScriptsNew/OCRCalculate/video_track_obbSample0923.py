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
from defect_infoSegment0923 import FrameData
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

                    process_id += 1  # 实际处理的帧数
                frame_id += 1  # 总帧计数器
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

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

    def get_all_frame_data(self):
        return self.frame_data_dict
