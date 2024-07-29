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
            'GL': Defect(),
            'FS': Defect(),
            'CJ': Defect(),
            'LD': Defect(),
            'LJ': Defect(),
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

    def calculate_fs_severity(self, rectangles):
        """
        计算腐蚀 (FS) 严重程度
        Args:
        rectangles (dict): 所有腐蚀的矩形数据
        Returns:
        int: 严重程度值（0-未知, 1-轻度, 2-中度, 3-重度）
        """
        # 计算所有腐蚀矩形的总面积
        total_area = sum(
            float(array1[0]) * float(array2[0])
            for array1, array2 in rectangles.values()
        )
        total_frame_area = self.video_width * self.video_height
        area_ratio = total_area / total_frame_area

        # 判断严重程度
        if area_ratio > 0.3:
            return 3
        elif area_ratio > 0.1:
            return 2
        else:
            return 1

    def calculate_gl_severity(self, rectangles):
        """
        计算管瘤 (GL) 严重程度
        """
        total_area = sum(
            float(array1[0]) * float(array2[0])
            for array1, array2 in rectangles.values()
        )
        total_frame_area = self.video_width * self.video_height
        area_ratio = total_area / total_frame_area

        if area_ratio > 0.3:
            return 3
        elif area_ratio > 0.1:
            return 2
        else:
            return 1

    def calculate_qn_severity(self, width, height, start_length):
        """
        计算气囊 (QN) 严重程度
        """
        if width > 0.3 * self.video_width or height > 0.3 * self.video_height or start_length > 5:
            return 3
        elif width > 0.1 * self.video_width or height > 0.1 * self.video_height or start_length > 3:
            return 2
        else:
            return 1

    def calculate_cj_severity(self, width, height, start_length):
        """
        计算沉积 (CJ) 严重程度
        """
        if width > 0.3 * self.video_width or height > 0.3 * self.video_height or start_length > 10:
            return 3
        elif width > 0.15 * self.video_width or height > 0.15 * self.video_height or start_length > 5:
            return 2
        else:
            return 1

    def calculate_yw_severity(self, width, height, start_length):
        """
        计算异物 (YW) 严重程度
        """
        if width > 0.3 * self.video_width or height > 0.3 * self.video_height or start_length > 10:
            return 3
        elif width > 0.15 * self.video_width or height > 0.15 * self.video_height or start_length > 5:
            return 2
        else:
            return 1

    def calculate_severity(self, defect_type, width=None, height=None, start_length=None, rectangles=None):
        """
        根据缺陷类型、宽、高等计算严重程度
        """
        if defect_type == "FS":
            return self.calculate_fs_severity(rectangles)
        elif defect_type == "GL":
            return self.calculate_gl_severity(rectangles)
        elif defect_type == "QN":
            return self.calculate_qn_severity(width, height, start_length)
        elif defect_type == "CJ":
            return self.calculate_cj_severity(width, height, start_length)
        elif defect_type == "YW":
            return self.calculate_yw_severity(width, height, start_length)
        else:
            return 0


class DefectInfoProcessor:
    def __init__(self, video_width, video_height, frame_window=5):
        """
        初始化缺陷信息处理器
        Args:
        frame_window (int): 连续多少帧未出现即视为缺陷结束
        """
        self.frame_window = frame_window
        self.defect_infos: Dict[str, DefectInfo] = {}
        self.active_defects = defaultdict(lambda: {"last_frame": 0, "start_distance": None, "severity": 0})
        self.severity_calculator = DefectSeverityCalculator(video_width, video_height)

    def update_defect_info(self, frame_data: FrameData):
        """
        更新缺陷信息
        Args:
        frame_data (FrameData): 包含当前帧数据的 FrameData 对象
        """
        frame_id = frame_data.frame_id
        distance = frame_data.distance

        for defect_type, defect in frame_data.defects.items():
            if defect_type in ["FS", "GL"]:
                # 对于腐蚀和管瘤，计算总面积
                severity = self.severity_calculator.calculate_severity(
                    defect_type,
                    rectangles=defect.rectangles
                )
                for defect_id in defect.rectangles.keys():
                    key = f"{defect_type}_{defect_id}"
                    if key not in self.active_defects:
                        self.active_defects[key] = {
                            "last_frame": frame_id,
                            "start_frame": frame_id,
                            "start_distance": distance,
                            "severity": severity
                        }
                    else:
                        self.active_defects[key]["last_frame"] = frame_id

            else:
                for defect_id, (array1, array2) in defect.rectangles.items():
                    # 检查 `array1` 是否为数组或元组
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
                    # 计算缺陷的长度（从开始位置到当前帧的位置）
                    start_length = distance - self.active_defects.get(f"{defect_type}_{defect_id}", {}).get(
                        "start_distance", distance)

                    severity = self.severity_calculator.calculate_severity(
                        defect_type,
                        width=width,
                        height=height,
                        start_length=start_length
                    )

                    key = f"{defect_type}_{defect_id}"
                    if key not in self.active_defects:
                        self.active_defects[key] = {
                            "last_frame": frame_id,
                            "start_frame": frame_id,
                            "start_distance": distance,
                            "severity": severity
                        }
                    else:
                        self.active_defects[key]["last_frame"] = frame_id
                        self.active_defects[key]["severity"] = severity

        # 检查并更新结束位置
        to_remove = []
        for key, info in self.active_defects.items():
            if frame_id - info["last_frame"] > self.frame_window:
                start_frame = info["start_frame"]
                start_distance = info["start_distance"]
                end_frame = info["last_frame"]
                end_distance = distance
                defect_type, defect_id = key.split("_")

                self.defect_infos[key] = DefectInfo(
                    name=key,
                    defect_type=defect_type,
                    defect_id=defect_id,
                    start_position=start_distance,
                    end_position=end_distance,
                    severity=info["severity"]
                )
                to_remove.append(key)

        # 从活动缺陷列表中移除已完成的缺陷
        for key in to_remove:
            del self.active_defects[key]

    def update_defect_infotest(self, frame_data: FrameData):
        """
        更新缺陷信息
        Args:
        frame_data (FrameData): 包含当前帧数据的 FrameData 对象
        """
        frame_id = frame_data.frame_id
        distance = frame_data.distance

        for defect_type, defect in frame_data.defects.items():
            for defect_id, (array1, array2) in defect.rectangles.items():
                # 根据数组结构提取具体的宽度和高度
                width = array1[0]  # 假设第一个数组的第一个元素是宽度
                height = array2[0]  # 假设第二个数组的第一个元素是高度
                key = f"{defect_type}_{defect_id}"
                if key not in self.active_defects:
                    self.active_defects[key] = {
                        "last_frame": frame_data.frame_id,
                        "start_distance": frame_data.distance,
                        "severity": self.severity_calculator.calculate_severity(defect_type, width, height)
                    }
                else:
                    self.active_defects[key]["last_frame"] = frame_data.frame_id

        # 检查并更新结束位置
        to_remove = []
        for key, info in self.active_defects.items():
            if frame_id - info["last_frame"] > self.frame_window:
                start_position = info["start_distance"]
                end_position = distance
                defect_type, defect_id = key.split("_")

                self.defect_infos[key] = DefectInfo(
                    name=key,
                    defect_type=defect_type,
                    defect_id=defect_id,
                    start_position=start_position,
                    end_position=end_position,
                    severity=info["severity"]
                )
                to_remove.append(key)

        # 从活动缺陷列表中移除已完成的缺陷
        for key in to_remove:
            del self.active_defects[key]
    def filter_defects(self, all_frame_data):
        valid_defects = {}
        temp_defects = defaultdict(lambda: defaultdict(list))

        # 记录每个缺陷在连续10帧内的出现次数
        for frame_id, frame_data in all_frame_data.items():
            for defect_type, defect in frame_data.defects.items():
                for defect_id in defect.rectangles:
                    temp_defects[defect_type][defect_id].append(frame_id)

        # 过滤掉不符合条件的缺陷
        for defect_type, defects in temp_defects.items():
            for defect_id, frames in defects.items():
                if len(frames) >= 5 and (max(frames) - min(frames)) <= 10:
                    valid_defects[defect_id] = frames[-1]  # 取最后一帧作为有效数据

        # 构建新的 all_frame_data
        new_frame_data = {}
        for frame_id, frame_data in all_frame_data.items():
            new_defects = {}
            for defect_type, defect in frame_data.defects.items():
                new_rectangles = {defect_id: defect.rectangles[defect_id]
                                  for defect_id in defect.rectangles
                                  if defect_id in valid_defects and valid_defects[defect_id] == frame_id}
                if new_rectangles:
                    new_defects[defect_type] = Defect()
                    new_defects[defect_type].rectangles = new_rectangles
            if new_defects:
                new_frame_data[frame_id] = FrameData(frame_id)
                new_frame_data[frame_id].defects = new_defects

        return new_frame_data

    def process_all_frame_data(self, all_frame_data):
            print("Processing all_frame_data...")
            # 检查并打印每个 FrameData 对象
            for i, frame_data in all_frame_data.items():
                if not isinstance(frame_data, FrameData):
                    print(f"Element at index {i} is not a FrameData object, but {type(frame_data)}")
                else:
                    # 打印 FrameData 对象的属性
                    print(f"Element {i + 1}: Frame ID = {frame_data.frame_id}")
                    print(f"  Distance: {frame_data.distance}")

                    # 更新缺陷信息
                    self.update_defect_info(frame_data)

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
    SEVERITY_SCORES = {'轻度': 2, '中度': 5, '重度': 10, '未知': 0}
    SEVERITY_MAP = {0: '未知', 1: '轻度', 2: '中度', 3: '重度'}

    def __init__(self, defect_infos, total_length):
        self.defect_infos = defect_infos
        self.total_length = total_length
        self.pipeline_sections = {}
        self.functional_defect_counts = defaultdict(int)

    def calculate_section_severity(self):
        for defect in self.defect_infos:
            start = defect['起始位置']
            end = defect['结束位置']
            defect_type = defect['类型']
            severity = defect['严重程度']

            if defect_type in ['FS', 'GL']:  # Structural defects
                key = (defect_type, start, end)
                if key not in self.pipeline_sections:
                    self.pipeline_sections[key] = {
                        'counts': {'轻度': 0, '中度': 0, '重度': 0},
                        'length': end - start
                    }
                self.pipeline_sections[key]['counts'][severity] += 1
            else:  # Functional defects
                self.functional_defect_counts[severity] += 1

    def determine_final_severity(self):
        for key, section in self.pipeline_sections.items():
            counts = section['counts']
            if counts['重度'] >= 1 or (counts['中度'] >= 1 and sum(counts.values()) > 5):
                final_severity = '重度'
            elif counts['中度'] >= 1 or counts['轻度'] > 5:
                final_severity = '中度'
            elif counts['轻度'] >= 2:
                final_severity = '轻度'
            else:
                final_severity = '未知'
            section['severity'] = final_severity

    def calculate_functional_score(self):
        functional_score = sum(
            self.SEVERITY_SCORES[severity] * count for severity, count in self.functional_defect_counts.items()
        ) / max(1, sum(self.functional_defect_counts.values()))
        return functional_score

    def evaluate_defects(self):
        self.calculate_section_severity()
        self.determine_final_severity()

        structural_score = sum(
            self.SEVERITY_SCORES[section['severity']] * (section['length'] / self.total_length)
            for section in self.pipeline_sections.values()
        )
        functional_score = self.calculate_functional_score()

        return structural_score, functional_score

    def save_evaluation_to_json(self, file_path):
        structural_score, functional_score = self.evaluate_defects()
        results = {
            'Pipeline Sections': {
                f"{key[0]} from {key[1]}m to {key[2]}m": {
                    'counts': section['counts'],
                    'length': section['length'],
                    'severity': section['severity']
                } for key, section in self.pipeline_sections.items()
            },
            'Structural Score': structural_score,
            'Functional Score': functional_score
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    def print_summary(self):
        structural_score, functional_score = self.evaluate_defects()
        print(f"Structural Score: {structural_score}")
        print(f"Functional Score: {functional_score}")
        for section, details in self.pipeline_sections.items():
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

    def filter_and_merge_audio_frequencies(self, input_json_path, output_json_path, freq_threshold=4000.0, db_threshold=-110.0, duration_seconds=5, frame_window=200, max_gap_seconds=10):
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

            # 检查当前段落是否超过帧窗口或持续时间
            if active_segment and (frame_count > frame_window or current_time - active_segment['start_time'] > duration_seconds):
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
        all_frame_data (dict): 包含每一帧数据的字典
        audio_results (list): 音频分析结果的列表
        """
        current_id = 1

        for result in audio_results:
            start_frame = result['start_frame']
            end_frame = result['end_frame']

            # 获取开始和结束位置的距离
            start_distance = all_frame_data[start_frame].distance if start_frame in all_frame_data else None
            end_distance = all_frame_data[end_frame].distance if end_frame in all_frame_data else None

            if start_distance is not None and end_distance is not None:
                # 为 "LD" 类型创建或更新 FrameData 对象
                for frame_id in range(start_frame, end_frame + 1):
                    if frame_id in all_frame_data:
                        frame_data = all_frame_data[frame_id]
                        if "LD" not in frame_data.defects:
                            frame_data.defects["LD"] = Defect()
                        frame_data.defects["LD"].add_rectangle(current_id, start_distance, end_distance)

                current_id += 1