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

    def add_rectangle(self, defect_id, width, height):
        """为缺陷添加矩形尺寸"""
        if defect_id not in self.rectangles:
            self.rectangles[defect_id] = (width, height)
            self.increment_count()

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

    def add_defect(self, defect_type, defect_id, width, height):
        """添加或更新缺陷类型的矩形尺寸，仅当ID首次出现时"""
        if defect_type in self.defects:
            self.defects[defect_type].add_rectangle(defect_id, width, height)

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
        if width*height > 0.3 * self.video_width * self.video_height or start_length > 10:
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
    def __init__(self, video_width, video_height, frame_window=5, merge_distance=0.5):
        """
        初始化缺陷信息处理器
        Args:
        frame_window (int): 连续多少帧未出现即视为缺陷结束
        """
        self.frame_window = frame_window
        self.defect_infos: Dict[str, DefectInfo] = {}
        self.active_defects = defaultdict(lambda: {"last_frame": 0, "start_distance": None, "severity": 0})
        self.video_width = video_width
        self.video_height = video_height
        self.merge_distance = merge_distance
        self.severity_calculator = DefectSeverityCalculator(video_width, video_height)

    def map_defect_type_to_base(self, defect_type):
        """将具体的缺陷类型映射到其基本类型"""
        if defect_type in ["LFS", "MFS", "SFS", 'SSFS']:
            return "FS"
        elif defect_type in ["LGL", "MGL", "SGL", 'SSGL']:
            return "GL"
        return defect_type

    def filter_defects(self, all_frame_data, min_frames=10, max_gap=5, segment_gap=10):
        """
        过滤并更新缺陷信息，保留满足连续性条件的缺陷。

        Args:
            all_frame_data: 原始帧数据字典。
            min_frames (int): 缺陷出现的最少帧数。
            max_gap (int): 允许的最大帧间隔。
            segment_gap (int): 允许的缺陷分段最大间隔。
        """
        valid_defects = defaultdict(lambda: defaultdict(list))
        temp_defects = defaultdict(lambda: defaultdict(list))

        # 记录每个缺陷在每帧内的出现次数
        for frame_id, frame_data in all_frame_data.items():
            for defect_type, defect in frame_data.defects.items():
                for defect_id, dimensions in defect.rectangles.items():
                    if defect_id is not None and defect_type is not None:
                        temp_defects[defect_type][defect_id].append(frame_id)

        # 过滤并确定符合连续性条件的缺陷
        for defect_type, defects in temp_defects.items():
            for defect_id, frames in defects.items():
                if len(frames) >= min_frames and self.is_continuous(frames, max_gap, segment_gap):
                    valid_defects[defect_type][defect_id] = frames

        # 更新 all_frame_data 以仅包含连续的缺陷信息
        for frame_id, frame_data in all_frame_data.items():
            for defect_type in list(frame_data.defects.keys()):
                defect = frame_data.defects[defect_type]
                # 仅保留有效缺陷的矩形信息
                new_rectangles = {defect_id: dimensions for defect_id, dimensions in defect.rectangles.items()
                                  if defect_id in valid_defects[defect_type] and frame_id in valid_defects[defect_type][
                                      defect_id]}
                if new_rectangles:
                    frame_data.defects[defect_type].rectangles = new_rectangles
                else:
                    del frame_data.defects[defect_type]  # 删除没有有效缺陷的类型


    def is_continuous(self, frames, max_gap, segment_gap):
        """
        检查帧列表是否满足连续性要求。

        Args:
            frames (list): 帧ID列表。
            max_gap (int): 允许的最大帧间隔。
            segment_gap (int): 允许的缺陷分段最大间隔。

        Returns:
            bool: 是否满足连续性。
        """
        frames = sorted(frames)
        gaps = [frames[i] - frames[i - 1] for i in range(1, len(frames))]
        # 允许存在至多一个超过segment_gap的间隔
        return sum(1 for gap in gaps if gap > max_gap) <= 1 and max(gaps) <= segment_gap


    def update_frame_data(self, frame_data: FrameData):
        frame_id = frame_data.frame_id
        distance = frame_data.distance

        # 处理基于严重程度的缺陷类型
        severity_levels = {
            'LFS': 'FS', 'MFS': 'FS', 'SFS': 'FS', 'SSFS': 'FS',
            'LGL': 'GL', 'MGL': 'GL', 'SGL': 'GL', 'SSGL': 'GL'
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
                for defect_id, (array1, array2) in defect.rectangles.items():
                    # 确保正确解析宽度和高度
                    if isinstance(array1, (list, tuple, np.ndarray)):
                        width = float(array1[0])
                    else:
                        # 直接将 `array1` 作为浮点数
                        width = float(array1)

                    # 同样处理 `array2`
                    if isinstance(array2, (list, tuple, np.ndarray)):
                        height = float(array2[0])
                    else:
                        height = float(array2)

                    key = f"{base_defect_type}_{defect_id}"
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
                for defect_id, (array1, array2) in defect.rectangles.items():
                    if isinstance(array1, (list, tuple, np.ndarray)) and isinstance(array2, (list, tuple, np.ndarray)):
                        width = float(array1[0])
                        height = float(array2[0])
                    else:
                        continue  # 跳过这个缺陷，因为数据不完整

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

    def build_defect_infos(self, current_frame_id):
        """
        根据当前帧ID构建缺陷信息。
        """
        to_remove = []
        for key, info in self.active_defects.items():
            defect_type = key.split('_')[0]
            frame_window = self.custom_frame_window(defect_type)  # 根据缺陷类型获取特定的窗口大小

            if current_frame_id - info["last_frame"] >= frame_window:
                self.finalize_defect_info(key, info)

            # 检查是否达到了自定义的结束条件
            if self.is_defect_ended(info, current_frame_id):
                self.finalize_defect_info(key, info)
                to_remove.append(key)

        for key in to_remove:
            del self.active_defects[key]

    def custom_frame_window(self, defect_type):
        """
        返回针对不同缺陷类型定制的frame window大小。
        """
        default_window = 5
        type_specific_windows = {'LD': 0, 'QN': 0}  # 举例
        return type_specific_windows.get(defect_type, default_window)

    def is_defect_ended(self, defect_info, current_frame_id):
        """
        根据缺陷的信息和当前帧ID判断缺陷是否应该结束。
        """
        # 示例：检查跟踪时长是否超过最大限度
        return False

    def finalize_defect_info(self, key, info):
        """
        处理并存储最终的缺陷信息。
        """
        start_frame = info["start_frame"]
        start_distance = info["start_distance"]
        end_frame = info["last_frame"]
        end_distance = info["end_distance"]
        self.defect_infos[key] = DefectInfo(
            name=key.split('_')[1],
            defect_type=key.split('_')[0],
            defect_id=key,
            start_position=start_distance,
            end_position=end_distance,
            severity=info["severity"]
        )

    def merge_defects(self):
        # Sort defects by type, severity, and start position
        sorted_defects = sorted(self.defect_infos.values(), key=lambda x: (x.defect_type, x.severity, x.start_position))
        merged_defects = {}
        current_defect = None

        for defect in sorted_defects:
            if current_defect and defect.defect_type == current_defect.defect_type and defect.severity == current_defect.severity:
                if defect.start_position - current_defect.end_position <= self.merge_distance:
                    # Extend the current defect
                    current_defect.end_position = max(current_defect.end_position, defect.end_position)
                    current_defect.severity = self.update_severity_based_on_length(current_defect.defect_type,
                                                                                   current_defect.start_position,
                                                                                   current_defect.end_position,
                                                                                   current_defect.severity)
                else:
                    # Save the current defect and start a new one
                    merged_defects[current_defect.defect_id] = current_defect
                    current_defect = defect
            else:
                if current_defect:
                    merged_defects[current_defect.defect_id] = current_defect
                current_defect = defect

        if current_defect:
            merged_defects[current_defect.defect_id] = current_defect

        self.defect_infos = merged_defects

    def process_all_frame_data(self, all_frame_data):
        """
        处理所有帧数据。
        """
        for frame_id, frame_data in all_frame_data.items():
            self.update_frame_data(frame_data)

        for frame_id, frame_data in all_frame_data.items():
            self.build_defect_infos(frame_id)
        self.merge_defects()

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


class DefectPreprocessor:
    def __init__(self, defects, merge_distance=0.5):
        self.defects = defects
        self.merge_distance = merge_distance

    def filter_and_merge_defects(self):
        # Filter out irrelevant defect types
        relevant_defects = [d for d in self.defects if d['类型'] not in ['ZG', 'CJ']]

        # Merge defects based on type, severity, and proximity
        merged_defects = []
        for defect in sorted(relevant_defects, key=lambda x: (x['类型'], x['严重程度'], x['起始位置'])):
            if merged_defects and self.can_merge(merged_defects[-1], defect):
                merged_defects[-1]['结束位置'] = max(merged_defects[-1]['结束位置'], defect['结束位置'])
            else:
                merged_defects.append(defect)
        return merged_defects

    def can_merge(self, defect1, defect2):
        # Check if same type and severity and within merge distance
        return (defect1['类型'] == defect2['类型'] and
                defect1['严重程度'] == defect2['严重程度'] and
                (defect2['起始位置'] - defect1['结束位置']) <= self.merge_distance)

class PipelineDefectEvaluator:
    SEVERITY_SCORES = {'轻度': 2, '中度': 5, '重度': 10, '未知': 0}
    SEVERITY_MAP = {0: '未知', 1: '轻度', 2: '中度', 3: '重度'}
    LD_SCORE = 10  # 漏点固定分数

    def __init__(self, defect_infos, total_length):
        preprocessor = DefectPreprocessor(defect_infos)
        self.defect_infos = preprocessor.filter_and_merge_defects()
        self.total_length = abs(total_length)  # Ensure non-negative total length
        self.total_length = total_length
        self.pipeline_function_sections = {}
        self.pipeline_structure_sections = {}
        self.functional_defect_counts = defaultdict(int)
        self.has_ld = any(d['类型'] == 'LD' for d in self.defect_infos)  # 检查是否存在漏点

    def print_defect_info(self):
        for defect_info in self.defect_infos:
            print(defect_info)

    def merge_defects(self, defect_infos, max_gap=0.5):
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


    def merge_calculate_qn_severity(self):
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
                    merged_defects.append(current_defect)
                    current_defect = defect

        if current_defect:
            length = current_defect['结束位置'] - current_defect['起始位置']
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

    def merge_selective_defects(self):
        defect_types = ['GL', 'FS']  # 管瘤和腐蚀的类型
        new_defects = []

        # 将缺陷按类型和起始位置排序
        sorted_defects = sorted((d for d in self.defect_infos if d['类型'] in defect_types),
                                key=lambda x: (x['类型'], x['起始位置']))

        current_defect = None
        for defect in sorted_defects:
            if not current_defect:
                current_defect = defect
            else:
                # 检查是否连续或重叠，并且是相同类型
                if defect['类型'] == current_defect['类型'] and defect['起始位置'] <= current_defect['结束位置']:
                    # 选择严重程度最高的
                    if defect['严重程度'] > current_defect['严重程度']:
                        current_defect['严重程度'] = defect['严重程度']
                    current_defect['结束位置'] = max(current_defect['结束位置'], defect['结束位置'])
                else:
                    new_defects.append(current_defect)
                    current_defect = defect
        if current_defect:
            new_defects.append(current_defect)

        # 重新将其他缺陷添加回去
        new_defects.extend(d for d in self.defect_infos if d['类型'] not in defect_types)
        self.defect_infos = new_defects
        self.merge_calculate_qn_severity()

    def calculate_section_severity(self):
        for defect in self.defect_infos:
            start = defect['起始位置']
            end = defect['结束位置']
            defect_type = defect['类型']
            severity = defect['严重程度']

            if defect_type in ['FS', 'GL']:  # Structural defects
                key = (defect_type, start, end)
                if key not in self.pipeline_structure_sections:
                    self.pipeline_structure_sections[key] = {
                        'counts': {'轻度': 0, '中度': 0, '重度': 0,'未知': 0},
                        'length': abs(end - start)
                    }
                self.pipeline_structure_sections[key]['counts'][severity] += 1
            else:  # Functional defects
                key = (defect_type, start, end)
                if key not in self.pipeline_function_sections:
                    self.pipeline_function_sections[key] = {
                        'counts': {'轻度': 0, '中度': 0, '重度': 0,'未知': 0},
                        'length': abs(end - start)
                    }
                self.pipeline_function_sections[key]['counts'][severity] += 1
                self.functional_defect_counts[severity] += 1

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
                final_severity = '轻度'
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

    def calculate_functional_score(self, length_threshold=0.3):
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
                } for key, section in self.pipeline_function_sections.items()
            },
            'Structural Score': structural_score,
            'Functional Score': functional_score
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    def save_evaluation_to_csv(self, csv_file_path, length_threshold=0.3):
        structural_score, functional_score = self.evaluate_defects()

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
        db_threshold = mean_db + 0.668 * std_db

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

    def filter_and_merge_audio_frequencies(self, input_json_path, output_json_path, freq_threshold=4000.0, db_threshold=-135.0, duration_seconds=0, frame_window=0, max_gap_seconds=10):
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

    def filter_and_merge_audio_frequenciesnew(self, input_json_path, output_json_path, duration_seconds=5, frame_window=200,
                                           max_gap_seconds=10):
        freq_threshold, db_threshold = self.calculate_thresholds(input_json_path)

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
