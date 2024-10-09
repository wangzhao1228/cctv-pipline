import csv

import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import cv2
from collections import defaultdict
from typing import List
from numba.typed.typeddict import Dict


class Defect:
    def __init__(self):
        self.count = 0
        self.rectangles = {}  # 字典，key为缺陷ID，value为(宽, 高)

    def increment_count(self):
        """增加缺陷计数"""
        self.count += 1

    def add_rectangle(self, defect_id, width=0, height=0, x=0, y=0, rot=0, confidence=0):
        """为缺陷添加矩形尺寸和位置信息"""
        # 如果是新的缺陷ID，直接添加
        if defect_id not in self.rectangles:
            self.rectangles[defect_id] = (width, height, x, y, rot, confidence)
            self.increment_count()
        else:
            # 如果已存在，检查是否需要更新（基于置信度或其他标准）
            existing_rect = self.rectangles[defect_id]
            # 可以根据需要调整合并条件，例如重叠区域、距离阈值或置信度
            if self.should_merge(existing_rect, (width, height, x, y, rot, confidence)):
                # 合并规则：取置信度更高的检测结果
                if confidence > existing_rect[4]:
                    self.rectangles[defect_id] = (width, height, x, y, rot, confidence)

    def should_merge(self, rect1, rect2):
        """判断两个矩形是否足够接近可以合并"""
        _, _, x1, y1, _, _ = rect1
        _, _, x2, y2, _, _ = rect2
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return distance < 50  # 示例阈值

    def to_dict(self):
        """将缺陷信息转换为字典格式"""
        return {
            "数量": self.count,
            "矩形尺寸": self.rectangles
        }


class FrameData:
    def __init__(self, frame_id=0):
        self.frame_id = frame_id
        self.distance = None  # 存储每帧的单个浮点距离数据
        self.defects = {
            'QN': Defect(),
            'YW': Defect(),
            'LGL': Defect(),
            'MGL': Defect(),
            'SGL': Defect(),
            'LFS': Defect(),
            'MFS': Defect(),
            'SFS': Defect(),
            'CJ': Defect(),
            'LD': Defect(),
            'ZG': Defect(),
            'TL': Defect(),
            'WZ': Defect(),
            'ZC': Defect(),
            'SZ': Defect(),
            'FM': Defect(),
            'PS': Defect(),
            'SSFS': Defect(),
            'SSGL': Defect(),
        }

    def clear_data(self):
        """清空所有数据，准备新一帧的数据处理"""
        self.distance = None
        for defect in self.defects.values():
            defect.count = 0
            defect.rectangles = {}

    def add_defect(self, defect_type, defect_id, width, height, x_center=0, y_center=0, rot=0, conf=0):
        """添加或更新缺陷类型的矩形尺寸，仅当ID首次出现时"""
        if defect_type in self.defects:
            self.defects[defect_type].add_rectangle(defect_id, width, height, x_center, y_center, rot, conf)

    def set_distance(self, distance):
        """设置距离数据"""
        self.distance = distance

    def update_frame_id(self, frame_id):
        """更新当前帧的序号"""
        self.frame_id = frame_id

    def get_defect_info(self, defect_type):
        """获取指定缺陷的详细信息"""
        if defect_type in self.defects:
            return self.defects[defect_type].to_dict()
        else:
            return None

    def remove_defect(self, defect_type):
        """删除一个缺陷类型"""
        if defect_type in self.defects:
            del self.defects[defect_type]

    def to_dict(self):
        return {
            "frame_id": self.frame_id,
            "distance": self.distance,
            "defects": {defect_type: defect.to_dict() for defect_type, defect in self.defects.items()}
        }

class DefectInfo:
    SEVERITY_LEVELS = {
        0: "未知",
        1: "轻度",
        2: "中度",
        3: "重度"
    }

    def __init__(self, name, defect_type, defect_id, start_position, end_position, severity=0):
        """
        初始化缺陷信息
        Args:
        name (str): 缺陷的名称
        defect_type (str): 缺陷的类型
        defect_id (str): 缺陷的唯一标识
        start_position (int): 缺陷的起始位置
        end_position (int): 缺陷的结束位置
        severity (int): 缺陷的严重程度，默认值为0，表示未知
        """
        self.name = name
        self.defect_type = defect_type
        self.defect_id = defect_id
        self.start_position = start_position
        self.end_position = end_position
        self.severity = severity

    def update_severity(self, severity):
        """更新缺陷的严重程度"""
        if severity in DefectInfo.SEVERITY_LEVELS:
            self.severity = severity

    def to_dict(self):
        """将缺陷信息转换为字典格式"""
        return {
            "名称": self.name,
            "类型": self.defect_type,
            "ID": self.defect_id,
            "起始位置": self.start_position,
            "结束位置": self.end_position,
            "严重程度": DefectInfo.SEVERITY_LEVELS.get(self.severity, "未知")
        }

class DefectSeverityCalculator:
    """
    用于计算严重程度的工具类
    """
    def __init__(self, video_width, video_height):
        self.video_width = video_width
        self.video_height = video_height

    def calculate_qn_severity(self, width, height, start_length):
        """
        计算气囊 (QN) 严重程度
        """
        if width*height > 0.3 * self.video_width * self.video_height or start_length > 10:
            return 3
        elif width*height > 0.15 * self.video_width * self.video_height or start_length > 5:
            return 2
        else:
            return 1

    def calculate_cj_severity(self, width, height, start_length):
        """
        计算沉积 (CJ) 严重程度
        """
        if width*height > 0.3 * self.video_width * self.video_height or start_length > 10:
            return 3
        elif width*height > 0.15 * self.video_width * self.video_height or start_length > 5:
            return 2
        else:
            return 1

    def calculate_yw_severity(self, width, height, start_length):
        """
        计算异物 (YW) 严重程度
        """
        if width*height > 0.4 * self.video_width * self.video_height or start_length > 10:
            return 3
        elif width*height > 0.15 * self.video_width * self.video_height or start_length > 5:
            return 2
        else:
            return 1

    def calculate_severity(self, defect_type, width=None, height=None, start_length=None, rectangles=None):
        """
        根据缺陷类型、宽、高等计算严重程度
        """
        if defect_type == "QN":
            return self.calculate_qn_severity(width, height, start_length)
        elif defect_type == "CJ":
            return self.calculate_cj_severity(width, height, start_length)
        elif defect_type == "YW":
            return self.calculate_yw_severity(width, height, start_length)
        elif defect_type == "LD":
            return 3
        else:
            return 1


class DefectInfoProcessor:
    def __init__(self, video_width, video_height, frame_window=5, merge_distance=0.2):
        """
        初始化缺陷信息处理器
        Args:
        frame_window (int): 连续多少帧未出现即视为缺陷结束
        """
        self.frame_window = frame_window
        self.valid_frames = {}
        self.defect_infos: Dict[str, DefectInfo] = {}
        self.active_defects = defaultdict(lambda: {"last_frame": 0, "start_distance": None, "severity": 0})
        self.video_width = video_width
        self.video_height = video_height
        self.merge_distance = merge_distance
        self.severity_calculator = DefectSeverityCalculator(video_width, video_height)
        # 新增缺陷类型分类
        self.functional_defects = {'YW', 'WZ', 'TL', 'QN', 'CJ', 'PS', 'FM', 'SZ', 'ZG', 'ZJ'}
        self.structural_defects = {'SSFS', 'SFS', 'MFS', 'LFS', 'SSGL', 'SGL', 'MGL', 'LGL', 'ZC'}

    # 根据缺陷的ID对缺陷进行分块
    def segment_and_update_defect_ids(self, all_frame_data, max_segment_gap=3):
        # 字典用于存储每一个缺陷实例的最后一帧和ID
        defect_dicts = {}

        for frame_id, frame_data in all_frame_data.items():
            for defect_type, defect in frame_data.defects.items():
                new_rectangles = {}
                if defect_type == 'ZC':
                    max_segment_gap = 3
                else:
                    max_segment_gap = 5
                for defect_id, dimensions in defect.rectangles.items():
                    # 定义键为 (defect_type, defect_id)
                    key = (defect_type, defect_id)
                    # 确保每个缺陷实例都有对应的记录
                    if key not in defect_dicts:
                        defect_dicts[key] = {'last_frame': frame_id, 'new_defect_id': 1}
                    else:
                        # 检查当前帧与最后记录帧的间隔
                        if frame_id - defect_dicts[key]['last_frame'] >= max_segment_gap:
                            # 如果间隔过大，认为是新的缺陷开始，增加新的缺陷ID
                            defect_dicts[key]['new_defect_id'] += 1

                    # 更新最后一帧为当前帧
                    defect_dicts[key]['last_frame'] = frame_id

                    # 生成新的缺陷ID
                    new_defect_id = defect_dicts[key]['new_defect_id']
                    new_rectangles[f"{defect_id}_{new_defect_id}"] = dimensions  # 使用原始缺陷ID和新的编号

                # 更新当前帧的缺陷数据
                defect.rectangles = new_rectangles

        return all_frame_data

    # 滤除不合适的framedata来构造defect_infos
    def filter_frames(self, all_frame_data, min_frames=3, max_gap=5, segment_gap=10):
        # temp of all_frame_data about same defect_id same defect_type
        temp_defect_frame = defaultdict(lambda: defaultdict(list))
        # vaild of all_frame_data
        self.valid_frames = defaultdict(lambda: defaultdict(list))

        self.segment_and_update_defect_ids(all_frame_data)
        # all_frame_data = self.segment_and_update_defect_ids(all_frame_data)

        for frame_id, frame_data in all_frame_data.items():
            for defect_type, defect in frame_data.defects.items():
                for defect_id, dimensions in defect.rectangles.items():
                    temp_defect_frame[defect_type][defect_id].append(frame_id)

        for defect_type, defects in temp_defect_frame.items():
            for defect_id, frames in defects.items():
                if defect_type in self.functional_defects:
                    if len(frames) >= min_frames:  # and self.is_continuous(frames, max_gap, segment_gap):  # 更宽松的连续性检查
                        self.valid_frames[defect_type][defect_id] = frames
                elif defect_type in self.structural_defects:
                    if len(frames) >= min_frames:  # and self.is_continuous(frames, max_gap, segment_gap):  # 更宽松的连续性检查
                        self.valid_frames[defect_type][defect_id] = frames

        # 打印出valid_defects的内容
        print("Valid Defects 01:")
        for defect_type, defects in self.valid_frames.items():
            print(f"Defect Type: {defect_type}")
            for defect_id, frames in defects.items():
                print(f"  Defect ID: {defect_id}, Frames: {frames}")

        # self.delete_unactivateframe_data(all_frame_data, self.valid_frames)

    # def delete_unactivateframe_data(self, all_frame_data, valid_defects):
    #     for frame_id, frame_data in all_frame_data.items():
    #         # 复制一份defect类型列表，避免在迭代中修改
    #         for defect_type in list(frame_data.defects.keys()):
    #             # 检查defect_type是否存在于valid_defects中
    #             if defect_type in valid_defects:
    #                 defect = frame_data.defects[defect_type]
    #                 new_rectangles = {}
    #
    #                 # 遍历当前缺陷类型的所有缺陷ID和对应的尺寸信息
    #                 for defect_id, dimensions in defect.rectangles.items():
    #                     # 检查当前的缺陷ID和当前帧ID是否都在有效缺陷列表中
    #                     if defect_id in valid_defects[defect_type] and frame_id in valid_defects[defect_type][defect_id]:
    #                         new_rectangles[defect_id] = dimensions
    #
    #                 # 如果有有效的矩形数据，更新当前帧中的缺陷数据
    #                 if new_rectangles:
    #                     frame_data.defects[defect_type].rectangles = new_rectangles
    #                 else:
    #                     # 如果没有有效的矩形数据，从当前帧中删除这个缺陷类型
    #                     del frame_data.defects[defect_type]
    #             else:
    #                 # 如果当前的defect_type不在有效缺陷类型中，直接删除
    #                 del frame_data.defects[defect_type]

    def is_continuous(self, frames, max_gap, segment_gap):
        frames.sort()
        gaps = [frames[i] - frames[i - 1] for i in range(1, len(frames))]
        return sum(1 for gap in gaps if gap > max_gap) <= 1 and max(gaps) <= segment_gap


    def bulid_active_defect(self, frame_data: FrameData):
        frame_id = frame_data.frame_id
        distance = frame_data.distance

        # 处理基于严重程度的缺陷类型
        severity_levels = {
            'LFS': 'FS', 'MFS': 'FS', 'SFS': 'FS', 'SSFS': 'FS',
            'LGL': 'GL', 'MGL': 'GL', 'SGL': 'GL', 'SSGL': 'GL'
        }
        severity_id_map = {
            'LFS': 40000, 'MFS': 30000, 'SFS': 20000, 'SSFS': 10000,
            'LGL': 40000, 'MGL': 30000, 'SGL': 20000, 'SSGL': 10000
        }
        severity_map = {
            'LFS': 3, 'MFS': 2, 'SFS': 1, 'SSFS': 1,
            'LGL': 3, 'MGL': 2, 'SGL': 1, 'SSGL': 1
        }

        for defect_type, defect in frame_data.defects.items():
            if defect_type in severity_levels:
                # 这些缺陷类型需要特别处理严重程度
                base_defect_type = severity_levels[defect_type]
                severity = severity_map[defect_type]
                for defect_id, rectangles in defect.rectangles.items():
                    width, height, x, y, rot, confidence = rectangles
                    try:
                        defect_id = int(defect_id)  # 转换为整数
                    except ValueError:
                        defect_id= int(float(defect_id))  # 处理浮点数字符串
                    new_defect_id = severity_id_map[defect_type] + defect_id
                    key = f"{base_defect_type}_{new_defect_id}"
                    if key not in self.active_defects:
                        self.active_defects[key] = {
                            "last_frame": frame_id,
                            "start_frame": frame_id,
                            "start_distance": distance,
                            "end_distance": distance,
                            "severity": severity
                        }
                    else:
                        self.active_defects[key]["last_frame"] = frame_id
                        self.active_defects[key]["end_distance"] = distance
            elif defect_type == 'LD':
                for defect_id, (array1, array2) in defect.rectangles.items():
                    # 检查是否存在开始长度的计算
                    start_length = distance - self.active_defects.get(f"{defect_type}_{defect_id}", {}).get(
                        "start_distance", distance)
                    if start_length is None:
                        continue

                    severity = self.severity_calculator.calculate_severity(defect_type, 0, 0, start_length)

                    key = f"{defect_type}_{defect_id}"
                    if key not in self.active_defects:
                        self.active_defects[key] = {
                            "last_frame": frame_id,
                            "start_frame": frame_id,
                            "start_distance": distance,
                            "end_distance": distance,
                            "severity": severity
                        }
                    else:
                        self.active_defects[key]["last_frame"] = frame_id
                        self.active_defects[key]["end_distance"] = distance
            else:
                # 其他类型的缺陷按常规方式处理
                for defect_id, rectangle in defect.rectangles.items():
                    width, height, x, y, rot, confidence = rectangle
                    # if isinstance(array1, (list, tuple, np.ndarray)) and isinstance(array2, (list, tuple, np.ndarray)):
                    #     width = float(array1[0])
                    #     height = float(array2[0])
                    # else:
                    #     continue  # 跳过这个缺陷，因为数据不完整

                    # 检查 width 和 height 是否有效
                    if width is None or height is None:
                        continue

                    # 检查是否存在开始长度的计算
                    start_length = distance - self.active_defects.get(f"{defect_type}_{defect_id}", {}).get(
                        "start_distance", distance)
                    if start_length is None:
                        continue

                    severity = self.severity_calculator.calculate_severity(defect_type, width, height, start_length)

                    key = f"{defect_type}_{defect_id}"
                    if key not in self.active_defects:
                        self.active_defects[key] = {
                            "last_frame": frame_id,
                            "start_frame": frame_id,
                            "start_distance": distance,
                            "end_distance": distance,
                            "severity": severity
                        }
                    else:
                        self.active_defects[key]["last_frame"] = frame_id
                        self.active_defects[key]["end_distance"] = distance

    def update_severity_based_on_length(self, defect_type, start_distance, end_distance,originalseverity):
        length = abs(end_distance - start_distance)
        if defect_type == 'QN':  # 气囊
            if length > 5:
                return '重度'
            elif length > 3:
                return '中度'
            else:
                return '轻度'
        elif defect_type == 'CJ':  # 沉积
            if length > 10:
                return '重度'
            elif length > 5:
                return '中度'
            else:
                return '轻度'
        elif defect_type == 'YW':  # 异物
            if length > 10:
                return '重度'
            elif length > 5:
                return '中度'
            else:
                return '轻度'
        else:
            return originalseverity  # 未知类型或不适用于这种规则的类型

    def build_defect_infos(self, all_frame_data):
        """
        根据 valid_frames 构建缺陷信息 defect_infos
        """
        for defect_type, defects in self.valid_frames.items():
            for defect_id, frames in defects.items():
                frames.sort()
                start_frame = frames[0]
                end_frame = frames[-1]
                # 获取起始和结束位置的距离
                start_distance = all_frame_data[start_frame].distance
                end_distance = all_frame_data[end_frame].distance

                # 确保距离是有效的数字
                if start_distance is None or end_distance is None:
                    continue  # 如果缺少距离信息，跳过该缺陷

                # 确保起始距离小于结束距离
                if start_distance > end_distance:
                    start_distance, end_distance = end_distance, start_distance

                # 计算严重程度（可根据需求自定义计算方式）
                severity = self.calculate_defect_severity(defect_type, defect_id, frames, all_frame_data)

                # 构建缺陷ID
                key = f"{defect_type}_{defect_id}"

                # 创建并存储 DefectInfo 对象
                self.defect_infos[key] = DefectInfo(
                    name=defect_id,
                    defect_type=defect_type,
                    defect_id=key,
                    start_position=start_distance,
                    end_position=end_distance,
                    severity=severity
                )

    def calculate_defect_severity(self, defect_type, defect_id, frames, all_frame_data):
        """
        根据缺陷类型和帧数据计算缺陷的严重程度
        """
        # 示例：对于功能性缺陷，严重程度可能根据持续时间或面积计算
        if defect_type in ['QN', 'CJ', 'YW']:
            # 获取缺陷的持续长度
            start_distance = all_frame_data[frames[0]].distance
            end_distance = all_frame_data[frames[-1]].distance
            defect_length = abs(end_distance - start_distance)

            # 根据长度确定严重程度
            if defect_length > 10:
                return 3  # 重度
            elif defect_length > 5:
                return 2  # 中度
            else:
                return 1  # 轻度
        elif defect_type in ['LFS', 'MFS', 'SFS', 'SSFS', 'LGL', 'MGL', 'SGL', 'SSGL']:
            # 结构性缺陷的严重程度可能已经在 defect_type 中包含
            # 如前面处理的 'LFS', 'MFS', 'SFS' 等
            # 这里直接返回缺陷的严重程度
            severity_map = {
                'LFS': 3, 'MFS': 2, 'SFS': 1, 'SSFS': 1,
                'LGL': 3, 'MGL': 2, 'SGL': 1, 'SSGL': 1
            }
            # 假设缺陷ID中包含了严重程度的信息
            severity = severity_map.get(defect_type, 1)
            return severity
        else:
            # 默认返回轻度
            return 1

    def custom_frame_window(self, defect_type):
        """
        返回针对不同缺陷类型定制的frame window大小。
        """
        default_window = 5
        type_specific_windows = {'LD': 0, 'QN': 0}  # 举例
        return type_specific_windows.get(defect_type, default_window)

    def merge_defects(self):
        # Sort defects by type, severity, and start position
        sorted_defects = sorted(self.defect_infos.values(), key=lambda x: (x.defect_type, x.severity, x.start_position))
        merged_defects = {}
        current_defect = None

    def process_all_frame_data(self, all_frame_data):
        """
        处理所有帧数据。
        """
        self.filter_frames(all_frame_data)
        self.build_defect_infos(all_frame_data)


    def get_defect_infos(self):
        # 打印更新后的 defect_infos
        print("Updated defect_infos:")
        for key, info in self.defect_infos.items():
            info_dict = info.to_dict()
            print(f"Key: {key}")
            print(f"  名称: {info_dict['名称']}")
            print(f"  类型: {info_dict['类型']}")
            print(f"  ID: {info_dict['ID']}")
            print(f"  起始位置: {info_dict['起始位置']}")
            print(f"  结束位置: {info_dict['结束位置']}")
            print(f"  严重程度: {info_dict['严重程度']}")
        """返回所有已处理的缺陷信息"""
        return [info.to_dict() for info in self.defect_infos.values()]

    def get_pipeline_length(self):
        """计算并返回覆盖所有缺陷的管道总长度。"""
        if not self.defect_infos:
            return 0  # 如果没有缺陷信息，则返回0

        min_start = float('inf')
        max_end = float('-inf')

        for defect_info in self.defect_infos.values():
            min_start = min(min_start, defect_info.start_position)
            max_end = max(max_end, defect_info.end_position)

        return max_end - min_start


class DamageScorer:
    """
    根据 defect_infos 统计缺陷并计算评分
    """
    SCORE_COEFFICIENTS = {
        'LD': 10,  # 漏点
        'FS': {3: 10, 2: 5, 1: 2},  # 腐蚀
        'GL': {3: 8, 2: 4, 1: 2},  # 管瘤
        'YW': {3: 6, 2: 3, 1: 1},  # 异物
        'QN': {3: 10, 2: 8, 1: 6},  # 气囊
        'CJ': {3: 8, 2: 6, 1: 1},  # 沉积
    }

    SEVERITY_MAP = {
        "重度": 3,
        "中度": 2,
        "轻度": 1,
        "未知": 0
    }

    def __init__(self, defect_infos, total_length):
        """
        初始化评分器
        Args:
        defect_infos (dict): 包含每个缺陷信息的字典
        total_length (float): 管道总长度
        """
        self.defect_infos = defect_infos
        self.total_length = max(total_length, 0)

    def calculate_length_coefficient(self, defect_type, defect_length):
        """
        计算长度系数
        Args:
        defect_type (str): 缺陷类型
        defect_length (float): 缺陷长度或计数
        Returns:
        float: 长度系数
        """
        if defect_type in ['FS', 'GL']:  # 按长度计算的类型
            if self.total_length < 100:
                return 0.5
            return defect_length / self.total_length
        elif defect_type in ['QN', 'CJ', 'YW']:  # 按个数计算的类型
            if self.total_length < 100:
                return 0.5
            return 50 / self.total_length
        return 0

    def calculate_score(self):
        """
        计算受损状态评分
        Returns:
        float: 最终评分
        """
        total_score = 0

        for defect in self.defect_infos:
            defect_type = defect['类型']
            severity_str = defect['严重程度']

            # 从字符串映射到整数
            severity = self.SEVERITY_MAP.get(severity_str, 0)

            # 调试信息：检查 defect_type 和 severity
            print(f"Calculating coefficient for Type: {defect_type}, Severity: {severity_str} ({severity})")

            # 从评分系数表中获取系数
            if defect_type in self.SCORE_COEFFICIENTS:
                if isinstance(self.SCORE_COEFFICIENTS[defect_type], dict):
                    coefficient = self.SCORE_COEFFICIENTS[defect_type].get(severity, 0)
                else:
                    coefficient = self.SCORE_COEFFICIENTS[defect_type]
            else:
                coefficient = 0

            # 特殊处理：如果是漏点（LD）、重度腐蚀或重度气囊，则直接给满分
            if defect_type == 'LD' or (defect_type in ['FS', 'QN'] and severity == 3):
                total_score += 10
            else:
                defect_length = defect['结束位置'] - defect['起始位置']
                # 计算长度系数
                length_coefficient = self.calculate_length_coefficient(defect_type, defect_length)

                # 计算评分并累加
                total_score += coefficient * length_coefficient

        return total_score

    def summarize_defects(self):
        """
        统计管道内各缺陷的详细信息，并打印格式化输出
        Returns:
        dict: 各缺陷类型及其轻、中、重度的统计
        """
        summary = {
            'LD': {'数量': 0, '位置': []},  # 漏点
            'FS': {'轻度': 0, '中度': 0, '重度': 0, '总长': 0},  # 腐蚀
            'GL': {'轻度': 0, '中度': 0, '重度': 0, '总长': 0},  # 管瘤
            'QN': {'轻度': 0, '中度': 0, '重度': 0, '总数': 0},  # 气囊
            'YW': {'轻度': 0, '中度': 0, '重度': 0, '总数': 0},  # 异物
            'CJ': {'轻度': 0, '中度': 0, '重度': 0, '总数': 0},  # 沉积
            'ZG': {'轻度': 0, '中度': 0, '重度': 0, '总数': 0},  # 支管
        }

        for defect in self.defect_infos:
            defect_type = defect['类型']
            severity_str = defect['严重程度']

            if defect_type == 'LD':  # 漏点单独处理
                start_position = defect['起始位置']
                end_position = defect['结束位置']
                summary['LD']['数量'] += 1
                summary['LD']['位置'].append({
                    '起始位置': start_position,
                    '结束位置': end_position
                })

            elif defect_type in ['FS', 'GL']:  # 按长度统计
                defect_length = defect['结束位置'] - defect['起始位置']
                if severity_str in summary[defect_type]:
                    summary[defect_type][severity_str] += defect_length
                    summary[defect_type]['总长'] += defect_length

            elif defect_type in ['QN', 'YW', 'CJ']:  # 按个数统计
                if severity_str in summary[defect_type]:
                    summary[defect_type][severity_str] += 1
                    summary[defect_type]['总数'] += 1

        return summary

    def print_summary(self):
        """
        打印缺陷的统计信息
        """
        summary = self.summarize_defects()

        # 打印每种缺陷的统计信息
        for defect_type, data in summary.items():
            if defect_type == 'LD':
                positions = '; '.join(
                    [f"第{i + 1}处: 起始位置：{pos['起始位置']}米，结束位置：{pos['结束位置']}米" for i, pos in
                     enumerate(data['位置'])])
                print(f"{defect_type}: {{数量：{data['数量']}处; {positions}}}")
            else:
                stats = ', '.join([f"{severity}: {count}" for severity, count in data.items()])
                print(f"{defect_type}: {{{stats}}}")



class PipelineDefectEvaluator:
    SEVERITY_SCORES = {'轻度': 2, '中度': 5, '重度': 10, '未知': 2}
    SEVERITY_MAP = {0: '未知', 1: '轻度', 2: '中度', 3: '重度'}
    LD_SCORE = 10  # 漏点固定分数

    def __init__(self, defect_infos, total_length):
        self.defect_infos = defect_infos
        self.total_length = abs(total_length)  # Ensure non-negative total length
        self.total_length = total_length
        self.pipeline_function_sections = {}
        self.pipeline_structure_sections = {}
        self.functional_defect_counts = defaultdict(int)
        self.has_ld = any(d['类型'] == 'LD' for d in self.defect_infos)  # 检查是否存在漏点
        self.priority_map = {
            'ZC_轻度': 1.5,
            'GL_重度': 6, 'FS_重度': 5,
            'GL_中度': 4, 'FS_中度': 3,
            'GL_轻度': 2, 'FS_轻度': 1
        }

    def print_defect_info(self):
        for defect_info in self.defect_infos:
            print(defect_info)

    def merge_defects(self, defect_infos, max_gap=0.1):
        # 先按类型和严重程度对缺陷进行分组
        grouped_defects = defaultdict(list)
        for defect in defect_infos:
            key = (defect['类型'], defect['严重程度'])
            grouped_defects[key].append(defect)

        # 合并每个分组中的缺陷
        merged_defects = []
        for key, defects in grouped_defects.items():
            # 根据起始位置对缺陷进行排序
            defects.sort(key=lambda x: x['起始位置'])
            current_merge = defects[0]

            for defect in defects[1:]:
                # 检查两个缺陷的实际距离是否小于最大间隔
                if abs(defect['起始位置'] - current_merge['结束位置']) <= max_gap:
                    # 如果距离小于等于阈值，则合并缺陷
                    current_merge['结束位置'] = max(current_merge['结束位置'], defect['结束位置'])
                    current_merge['起始位置'] = min(current_merge['起始位置'], defect['起始位置'])
                else:
                    # 如果不连续，先保存当前段落，然后开始新的合并段落
                    merged_defects.append(current_merge)
                    current_merge = defect

            # 添加最后一个合并段
            merged_defects.append(current_merge)

        return merged_defects


    def merge_calculate_qn_severity(self,qn_length=0.1):
        qn_defects = sorted((d for d in self.defect_infos if d['类型'] == 'QN'),
                            key=lambda x: x['起始位置'])
        merged_defects = []
        current_defect = None

        for defect in qn_defects:
            if not current_defect:
                current_defect = defect
            else:
                if defect['起始位置'] <= current_defect['结束位置']+2:
                    current_defect['结束位置'] = max(current_defect['结束位置'], defect['结束位置'])
                    # 选择严重程度最高的
                    if defect['严重程度'] > current_defect['严重程度']:
                        current_defect['严重程度'] = defect['严重程度']
                else:
                    # 重新计算严重程度
                    length = current_defect['结束位置'] - current_defect['起始位置']
                    current_defect['严重程度'] = self.calculate_new_severity(length)
                    if length >= qn_length:  # 检查长度是否符合要求
                        merged_defects.append(current_defect)
                        current_defect = defect

        if current_defect:
            length = current_defect['结束位置'] - current_defect['起始位置']
            if length >= qn_length:  # 检查长度是否符合要求
                current_defect['严重程度'] = self.calculate_new_severity(length)
                merged_defects.append(current_defect)

        self.defect_infos = [d for d in self.defect_infos if d['类型'] != 'QN'] + merged_defects

    def calculate_new_severity(self, length):
        if length > 10:
            return '重度'
        elif length > 5:
            return '中度'
        else:
            return '轻度'

    def severity_score(self, severity):
        # 为不同的严重程度定义权重，方便排序和比较
        return {'轻度': 1, '中度': 2, '重度': 3}[severity]

    def resolve_overlaps(self, defects):
        if not defects:
            return []

        defects.sort(key=lambda x: x['起始位置'])
        resolved_defects = [defects[0]]

        for current in defects[1:]:
            last = resolved_defects[-1]
            # 仅合并相同类型且严重程度较高或相同的缺陷
            if current['起始位置'] <= last['结束位置'] and current['类型'] == last['类型']:
                if current['严重程度'] >= last['严重程度']:
                    last['结束位置'] = max(last['结束位置'], current['结束位置'])
            else:
                resolved_defects.append(current)

        return resolved_defects

    def merge_selective_defects(self):
        target_defect_types = ['GL', 'FS', 'ZC']
        priority_map = {
            'ZC_轻度': 1.5,
            'GL_重度': 6, 'FS_重度': 5,
            'GL_中度': 4, 'FS_中度': 3,
            'GL_轻度': 2, 'FS_轻度': 1
        }
        unprocessed_defects = list(filter(lambda d: d['类型'] in target_defect_types, self.defect_infos))
        unprocessed_defects.sort(key=lambda x: (x['起始位置'], -priority_map[f"{x['类型']}_{x['严重程度']}"]))
        processed_defects = []

        for current in unprocessed_defects:
            if not processed_defects:
                processed_defects.append(current)
                continue

            last = processed_defects[-1]

            if current['起始位置']>last['结束位置']:
                # 没有重叠
                processed_defects.append(current)
            else:
                # 存在重叠
                if priority_map[f"{current['类型']}_{current['严重程度']}"]> priority_map[f"{last['类型']}_{last['严重程度']}"]:
                    #当前缺陷严重程度更高
                    if current['起始位置']>last['起始位置']:
                        #分割last缺陷，保留前半部分
                        earlier_part = last.copy()
                        earlier_part['结束位置'] = current['起始位置']
                        processed_defects[-1] = earlier_part
                        processed_defects.append(current)
                    else:
                        #current完全覆盖last的开始部分
                        processed_defects[-1] = current

                    # 更新current以可能包括last的延伸部分
                    if current['结束位置'] < last['结束位置']:
                        # 如果当前缺陷延伸超过last，更新当前去诶按的开始位置并在循环外添加
                        later_part = last.copy()
                        later_part['起始位置'] = current['结束位置']
                        processed_defects.append(later_part)
                else:
                    #当前缺陷严重程度不高，或相同
                    if current['结束位置'] > last['结束位置']:
                        #如果当前缺陷延伸超过last，更新当前缺陷的开始位置并在循环外添加
                        current['起始位置'] = last['结束位置']
                        processed_defects.append(current)

        # Combine processed defects with non-target type defects
        self.defect_infos = processed_defects + [d for d in self.defect_infos if d['类型'] not in target_defect_types]

        # Handle special defect types like QN
        self.merge_calculate_qn_severity()

    def calculate_section_severity(self):
        type_mapping = {
            'LFS': 'FS', 'MFS': 'FS', 'SFS': 'FS', 'SSFS': 'FS',
            'LGL': 'GL', 'MGL': 'GL', 'SGL': 'GL', 'SSGL': 'GL'
        }

        for defect in self.defect_infos:
            start = defect['起始位置']
            end = defect['结束位置']
            defect_type = defect['类型']
            severity = defect['严重程度']

            # 将具体缺陷类型映射到基本类型
            base_defect_type = type_mapping.get(defect_type, defect_type)

            if base_defect_type in ['FS', 'GL']:  # Structural defects
                key = (base_defect_type, start, end)
                if key not in self.pipeline_structure_sections:
                    self.pipeline_structure_sections[key] = {
                        'counts': {'轻度': 0, '中度': 0, '重度': 0, '未知': 0},
                        'length': abs(end - start)
                    }
                self.pipeline_structure_sections[key]['counts'][severity] += 1
            elif base_defect_type in ['YW', 'ZG', 'CJ', 'QN']:  # Functional defects
                key = (base_defect_type, start, end)
                if key not in self.pipeline_function_sections:
                    self.pipeline_function_sections[key] = {
                        'counts': {'轻度': 0, '中度': 0, '重度': 0, '未知': 0},
                        'length': abs(end - start)
                    }
                self.pipeline_function_sections[key]['counts'][severity] += 1
                self.functional_defect_counts[severity] += 1

    def print_sections(self):
        print("Pipeline Structure Sections:")
        for key, section in self.pipeline_structure_sections.items():
            print(f"Type: {key[0]}, Start: {key[1]}, End: {key[2]}, Length: {section['length']} meters")
            for severity, count in section['counts'].items():
                print(f"  {severity}: {count}")

        print("\nPipeline Function Sections:")
        for key, section in self.pipeline_function_sections.items():
            print(f"Type: {key[0]}, Start: {key[1]}, End: {key[2]}, Length: {section['length']} meters")
            for severity, count in section['counts'].items():
                print(f"  {severity}: {count}")


    def determine_final_severity(self):
        for key, section in self.pipeline_structure_sections.items():
            counts = section['counts']
            if counts['重度'] >= 1:
                final_severity = '重度'
            elif counts['中度'] >= 1:
                final_severity = '中度'
            elif counts['轻度'] >= 1:
                final_severity = '轻度'
            else:
                final_severity = '未知'  # 修改默认值为 '未知' 或根据需求调整
            section['severity'] = final_severity
        for key, section in self.pipeline_function_sections.items():
            counts = section['counts']
            if counts['重度'] >= 1:
                final_severity = '重度'
            elif counts['中度'] >= 1:
                final_severity = '中度'
            elif counts['轻度'] >= 1:
                final_severity = '轻度'
            else:
                final_severity = '轻度'
            section['severity'] = final_severity

    def calculate_functional_score(self, length_threshold=0):
        functional_score = 0
        total_counts = 0
        # 遍历功能性缺陷部分，只计算长度大于指定阈值的部分
        for section_id, section_info in self.pipeline_function_sections.items():
            if section_info['length'] >= length_threshold:
                for severity, count in section_info['counts'].items():
                    functional_score += self.SEVERITY_SCORES[severity] * count
                    total_counts += count

        if total_counts > 0:
            functional_score /= total_counts
        else:
            functional_score = 0  # 如果没有任何符合条件的部分，分数为0

        return functional_score

    def evaluate_defects(self):
        self.calculate_section_severity()
        self.determine_final_severity()

        structural_score = sum(
            self.SEVERITY_SCORES[section['severity']] * (section['length'] / self.total_length)
            for section in self.pipeline_structure_sections.values()
        )
        functional_score = self.calculate_functional_score()

        # 如果存在至少一个漏点，结构性得分增加10分
        if self.has_ld:
            structural_score += self.LD_SCORE

        return structural_score, functional_score

    def save_evaluation_to_json(self, file_path):
        structural_score, functional_score = self.evaluate_defects()
        results = {
            'Pipeline Sections': {
                f"{key[0]} from {key[1]}m to {key[2]}m": {
                    'counts': section['counts'],
                    'length': section['length'],
                    'severity': section['severity']
                } for key, section in self.pipeline_structure_sections.items()
            },
            'Functional Sections': {
                f"{key[0]} from {key[1]}m to {key[2]}m": {
                    'counts': section['counts'],
                    'length': section['length'],
                    'severity': section['severity']
                } for key, section in self.pipeline_function_sections.items()
            },
            'Structural Score': structural_score,
            'Functional Score': functional_score
        }
        # 确保文件所在的目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    def save_evaluation_to_csv(self, csv_file_path, length_threshold=0):
        structural_score, functional_score = self.evaluate_defects()
        directory = os.path.dirname(csv_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(
                ['Defect Type', 'Start Position (m)', 'End Position (m)', 'Severity', 'Length (m)', 'Count'])

            # 写入结构性缺陷数据
            for key, section in self.pipeline_structure_sections.items():
                defect_type = key[0]
                start_position = key[1]
                end_position = key[2]
                severity = section['severity']
                length = section['length']
                count = sum(section['counts'].values())  # 总缺陷计数
                writer.writerow([defect_type, start_position, end_position, severity, length, count])

            # 遍历功能性缺陷部分，只保存长度大于指定阈值的部分
            for key, section in self.pipeline_function_sections.items():
                if section['length'] >= length_threshold:
                    defect_type = key[0]
                    start_position = key[1]
                    end_position = key[2]
                    severity = section['severity']
                    length = section['length']
                    count = sum(section['counts'].values())  # 总缺陷计数
                    writer.writerow([defect_type, start_position, end_position, severity, length, count])

    def print_summary(self):
        structural_score, functional_score = self.evaluate_defects()
        print(f"Structural Score: {structural_score}")
        print(f"Functional Score: {functional_score}")
        for section, details in self.pipeline_function_sections.items():
            print(f"{section[0]} from {section[1]}m to {section[2]}m:")
            for severity, count in details['counts'].items():
                print(f"  {severity}: {count}")
            print(f"  Final Severity: {details['severity']}")
            print(f"  Section Length: {details['length']} meters")


class AudioAnalyzer:
    def __init__(self, audio_path, video_path, tmin=0, tmax=None, alpha=0.97):
        """
        初始化音频分析器
        Args:
        audio_path (str): 音频文件路径
        video_path (str): 视频文件路径
        tmin (int): 开始时间（秒）
        tmax (int): 结束时间（秒），默认为None表示到文件结束
        alpha (float): 预加重系数
        """
        self.audio_path = audio_path
        self.video_path = video_path
        self.waveform, self.sr = librosa.load(audio_path)
        self.alpha = alpha
        self.tmin = tmin
        self.tmax = tmax if tmax is not None else len(self.waveform) / self.sr
        self.emphasized_y = np.append(
            self.waveform[tmin * self.sr],
            self.waveform[tmin * self.sr + 1:int(self.tmax * self.sr)] - alpha * self.waveform[tmin * self.sr:int(self.tmax * self.sr) - 1]
        )
        self.n = int((self.tmax - tmin) * self.sr)
        self.video_fps = self.get_video_fps()

    def get_video_fps(self):
        """
        获取视频的帧率
        Returns:
        float: 视频的帧率
        """
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def audio_time_to_video_frame(self, time_sec):
        """
        将音频时间转换为视频帧序号
        Args:
        time_sec (float): 音频时间（秒）
        Returns:
        int: 视频帧序号
        """
        return int(time_sec * self.video_fps)

    def compute_frames(self, frame_size=0.025, frame_stride=0.01):
        """
        计算帧并加窗
        Args:
        frame_size (float): 每一帧的长度，单位为秒
        frame_stride (float): 相邻两帧的间隔，单位为秒
        Returns:
        numpy.ndarray: 分帧的音频信号
        int: 帧的长度
        int: 总帧数
        """
        frame_length, frame_step = int(round(self.sr * frame_size)), int(round(self.sr * frame_stride))
        signal_length = int((self.tmax - self.tmin) * self.sr)
        frame_num = int(np.ceil((signal_length - frame_length) / frame_step)) + 1
        pad_frame = (frame_num - 1) * frame_step + frame_length - signal_length
        pad_y = np.append(self.emphasized_y, np.zeros(pad_frame))

        indices = np.tile(np.arange(0, frame_length), (frame_num, 1)) + np.tile(
            np.arange(0, frame_num * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_y[indices]
        frames *= np.hamming(frame_length)

        return frames, frame_length, frame_num

    def save_audio_frequencies_to_json(self, json_path, frame_size=0.025, frame_stride=0.01):
        """
        将每一视频帧对应的音频频率和分贝值保存到JSON文件
        Args:
        json_path (str): JSON文件保存路径
        frame_size (float): 每一帧的长度，单位为秒
        frame_stride (float): 相邻两帧的间隔，单位为秒
        """
        # 获取目标目录
        directory = os.path.dirname(json_path)

        # 如果目标目录不存在，则创建它
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        frames, frame_length, frame_num = self.compute_frames(frame_size, frame_stride)
        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = mag_frames ** 2 / NFFT
        pow_frames = np.where(pow_frames == 0, np.finfo(float).eps, pow_frames)  # 替换零值
        freqs = np.linspace(0, self.sr / 2, NFFT // 2 + 1)

        time_indices = librosa.frames_to_time(range(frame_num), sr=self.sr, hop_length=int(frame_stride * self.sr))
        data = []
        for i, t in enumerate(time_indices):
            max_freq_idx = np.argmax(pow_frames[i][:len(freqs)])
            max_freq = freqs[max_freq_idx]
            max_db = 20 * np.log10(pow_frames[i][max_freq_idx])
            video_frame = self.audio_time_to_video_frame(t)
            data.append({
                "video_frame": video_frame,
                "time": round(t, 3),
                "frequency": round(max_freq, 2),
                "decibel": round(max_db, 2)
            })

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def frame_features(self, frame):
        # 短时能量
        energy = np.sum(frame ** 2)

        # 频谱熵
        spectrum = np.abs(np.fft.fft(frame))
        normalized_spectrum = spectrum / np.sum(spectrum)
        spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10))

        # 谐波比率
        harmonic = librosa.effects.harmonic(frame, margin=3)
        harmonic_energy = np.sum(harmonic ** 2)
        harmonic_ratio = harmonic_energy / (energy + 1e-10)

        return energy, spectral_entropy, harmonic_ratio

    def calculate_thresholds(self,input_json_path):
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        frequencies = [item['frequency'] for item in data]
        decibels = [item['decibel'] for item in data]

        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        mean_db = np.mean(decibels)
        std_db = np.std(decibels)

        freq_threshold = mean_freq + 0.668 * std_freq
        db_threshold = mean_db - std_db

        return freq_threshold, db_threshold

    def plot_frequency_and_power_distributions(self, input_json_path):
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        frequencies = [item['frequency'] for item in data]
        decibels = [item['decibel'] for item in data]
        frames = [item['video_frame'] for item in data]

        # 计算阈值
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        mean_db = np.mean(decibels)
        std_db = np.std(decibels)

        freq_threshold = mean_freq + 0.668 * std_freq
        db_threshold = mean_db + 0.668 * std_db

        # 绘制频率分布图
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(frames, frequencies, marker='o', linestyle='', alpha=0.7, label='Frequency')
        plt.axhline(y=freq_threshold, color='red', label=f'Freq Threshold: {freq_threshold:.2f} Hz')
        plt.title('Frequency Distribution Over Frames')
        plt.xlabel('Video Frame Number')
        plt.ylabel('Frequency (Hz)')
        plt.legend()

        # 绘制功率分布图
        plt.subplot(1, 2, 2)
        plt.plot(frames, decibels, marker='o', linestyle='', alpha=0.7, color='green', label='Decibel')
        plt.axhline(y=db_threshold, color='red', label=f'DB Threshold: {db_threshold:.2f} dB')
        plt.title('Power (Decibel) Distribution Over Frames')
        plt.xlabel('Video Frame Number')
        plt.ylabel('Decibel (dB)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_threshold_effects(self, input_json_path):
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        frequencies = [item['frequency'] for item in data]
        decibels = [item['decibel'] for item in data]
        frames = [item['video_frame'] for item in data]

        # 计算阈值
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        freq_threshold = mean_freq + 0.668 * std_freq  # ~75% confidence level

        mean_db = np.mean(decibels)
        std_db = np.std(decibels)
        db_threshold = mean_db + 0.668 * std_db  # ~75% confidence level

        # 判断哪些帧满足频率和功率的阈值
        freq_meets_threshold = np.array(frequencies) >= freq_threshold
        db_meets_threshold = np.array(decibels) >= db_threshold

        # 绘制频率和功率分布
        plt.figure(figsize=(14, 6))

        # 绘制频率分布
        plt.subplot(1, 2, 1)
        plt.scatter(frames, frequencies, color='gray', label='Below Threshold')
        plt.scatter(np.array(frames)[freq_meets_threshold], np.array(frequencies)[freq_meets_threshold], color='blue', label='Above Frequency Threshold')
        plt.axhline(y=freq_threshold, color='red', linestyle='--', label=f'Freq Threshold: {freq_threshold:.2f} Hz')
        plt.title('Frequency Distribution Over Frames')
        plt.xlabel('Video Frame Number')
        plt.ylabel('Frequency (Hz)')
        plt.legend()

        # 绘制功率分布
        plt.subplot(1, 2, 2)
        plt.scatter(frames, decibels, color='gray', label='Below Threshold')
        plt.scatter(np.array(frames)[db_meets_threshold], np.array(decibels)[db_meets_threshold], color='green', label='Above Decibel Threshold')
        plt.axhline(y=db_threshold, color='red', linestyle='--', label=f'DB Threshold: {db_threshold:.2f} dB')
        plt.title('Power (Decibel) Distribution Over Frames')
        plt.xlabel('Video Frame Number')
        plt.ylabel('Decibel (dB)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_combined_thresholds(self, input_json_path):
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        frequencies = [item['frequency'] for item in data]
        decibels = [item['decibel'] for item in data]
        frames = [item['video_frame'] for item in data]

        # 计算阈值
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        freq_threshold = mean_freq - std_freq

        mean_db = np.mean(decibels)
        std_db = np.std(decibels)
        db_threshold = mean_db - std_db

        # 确定各帧数据是否满足阈值
        freq_meets_threshold = np.array(frequencies) >= freq_threshold
        db_meets_threshold = np.array(decibels) >= db_threshold
        both_meet_threshold = freq_meets_threshold & db_meets_threshold

        plt.figure(figsize=(10, 6))
        # 绘制所有未达标的点
        plt.scatter(np.array(frames)[~both_meet_threshold], np.array(frequencies)[~both_meet_threshold], color='gray',
                    alpha=0.5, label='Below Thresholds')
        # 绘制满足频率阈值的点
        plt.scatter(np.array(frames)[freq_meets_threshold], np.array(frequencies)[freq_meets_threshold], color='blue',
                    alpha=0.7, label='Above Frequency Threshold')
        # 绘制满足分贝阈值的点
        plt.scatter(np.array(frames)[db_meets_threshold], np.array(frequencies)[db_meets_threshold], color='green',
                    alpha=0.5, label='Above Decibel Threshold')
        # 绘制同时满足两个阈值的点
        plt.scatter(np.array(frames)[both_meet_threshold], np.array(frequencies)[both_meet_threshold], color='red',
                    label='Above Both Thresholds')

        plt.axhline(y=freq_threshold, color='blue', linestyle='--',
                    label=f'Frequency Threshold: {freq_threshold:.2f} Hz')
        plt.axhline(y=db_threshold, color='green', linestyle='--', label=f'DB Threshold: {db_threshold:.2f} dB')

        plt.title('Combined Frequency and Decibel Thresholds Over Frames')
        plt.xlabel('Video Frame Number')
        plt.ylabel('Frequency (Hz) and Decibel (dB)')
        plt.legend()
        plt.show()

    def filter_and_merge_audio_frequenciesold(self, input_json_path, output_json_path, freq_threshold=4000.0, db_threshold=-95.0, duration_seconds=5, frame_window=200, max_gap_seconds=10):
        """
        根据提供的阈值筛选音频频率和分贝值，并合并无时间间隔的段落，保存为新的JSON文件
        Args:
        input_json_path (str): 输入的JSON文件路径
        output_json_path (str): 输出的JSON文件路径
        freq_threshold (float): 频率阈值（Hz）
        db_threshold (float): 分贝阈值（dB）
        duration_seconds (float): 持续时间阈值（秒）
        frame_window (int): 持续帧数窗口（默认为30帧）
        max_gap_seconds (float): 合并段落之间的最大时间间隔（秒）
        """
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []
        active_segment = None
        frame_count = 0
        current_time = None

        for i, item in enumerate(data):
            current_time = item['time']
            if item['frequency'] >= freq_threshold and item['decibel'] >= db_threshold:
                if active_segment is None:
                    active_segment = {
                        "start_frame": item['video_frame'],
                        "start_time": item['time']
                    }
                frame_count = 0
            else:
                frame_count += 1

            # 检查当前段落是否超过帧窗口和持续时间
            if active_segment and (frame_count > frame_window and current_time - active_segment['start_time'] > duration_seconds):
                if current_time - active_segment['start_time'] >= duration_seconds:
                    active_segment['end_frame'] = data[i - 1]['video_frame']
                    active_segment['end_time'] = data[i - 1]['time']
                    results.append(active_segment)
                active_segment = None
                frame_count = 0

        if active_segment and current_time - active_segment['start_time'] >= duration_seconds:
            active_segment['end_frame'] = data[-1]['video_frame']
            active_segment['end_time'] = data[-1]['time']
            results.append(active_segment)

        merged_results = []
        if results:
            current_segment = results[0]
            for next_segment in results[1:]:
                if next_segment['start_time'] - current_segment['end_time'] <= max_gap_seconds:
                    current_segment['end_frame'] = next_segment['end_frame']
                    current_segment['end_time'] = next_segment['end_time']
                else:
                    merged_results.append(current_segment)
                    current_segment = next_segment
            merged_results.append(current_segment)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(merged_results, f, indent=4, ensure_ascii=False)

    def filter_and_merge_audio_frequencies(self, input_json_path, output_json_path, duration_seconds=15,
                                           frame_window=200, max_gap_seconds=10):
        freq_threshold, db_threshold = self.calculate_thresholds(input_json_path)
        print(freq_threshold, db_threshold)
        freq_threshold = 4000  # 确保频率阈值为4kHz
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []
        active_segment = None
        frame_count = 0
        current_time = None

        for i, item in enumerate(data):
            current_time = item['time']
            if item['frequency'] >= freq_threshold and item['decibel'] >= db_threshold:
                if active_segment is None:
                    active_segment = {
                        "start_frame": item['video_frame'],
                        "start_time": item['time']
                    }
                frame_count = 0
            else:
                frame_count += 1

            if active_segment and (
                    frame_count > frame_window and current_time - active_segment['start_time'] > duration_seconds):
                if current_time - active_segment['start_time'] >= duration_seconds:
                    active_segment['end_frame'] = data[i - 1]['video_frame']
                    active_segment['end_time'] = data[i - 1]['time']
                    results.append(active_segment)
                active_segment = None
                frame_count = 0

        if active_segment and current_time - active_segment['start_time'] >= duration_seconds:
            active_segment['end_frame'] = data[-1]['video_frame']
            active_segment['end_time'] = data[-1]['time']
            results.append(active_segment)

        # 过滤掉持续时间小于15秒的段落
        filtered_results = [segment for segment in results if segment['end_time'] - segment['start_time'] >= 15]

        # 保留两个最长的段落
        filtered_results.sort(key=lambda x: x['end_time'] - x['start_time'], reverse=True)
        longest_segments = filtered_results[:2]

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(longest_segments, f, indent=4, ensure_ascii=False)


    def integrate_audio_to_all_frame_data(self, all_frame_data, audio_results):
        """
        将音频分析结果集成到 all_frame_data 中
        Args:
            all_frame_data (dict): 包含每一帧数据的FrameData对象字典
            audio_results (list): 音频分析结果的列表
        """
        current_id = 1  # 初始化缺陷ID

        for result in audio_results:
            start_frame = result['start_frame']
            end_frame = result['end_frame']

            # 遍历音频分析结果中的每一个时间段
            for frame_id in range(start_frame, end_frame + 1):
                if frame_id in all_frame_data:
                    frame_data = all_frame_data[frame_id]
                    rectangle = [[0, 0],
                                 [0, 0]]  # Placeholder rectangle dimensions
                    frame_data.add_defect("LD", current_id, rectangle[0], rectangle[1])

            current_id += 1  # 更新缺陷ID
