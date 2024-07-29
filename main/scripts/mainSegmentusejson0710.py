import json
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from frame_ocr_interpolate import FrameOCR
from video_track_obb import VideoTracker

from defect_infoSegment import DefectInfoProcessor, DamageScorer, PipelineDefectEvaluator, AudioAnalyzer, FrameData, Defect


def load_frame_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    all_frame_data = {}
    for frame_id, details in data.items():
        distance = details.get('Distance')
        defects_data = details.get('Defects', {})

        frame_data = FrameData(frame_id=int(frame_id))
        frame_data.set_distance(distance)  # 假设存在此方法来设置距离

        for defect_type, defect_details in defects_data.items():
            defect = Defect()  # 假设Defect类有一个无参数的构造函数
            defect.increment_count()  # 假设每次调用都是为了增加计数，可能需要根据实际逻辑调整
            for rect_id, rect_vals in defect_details.get('Rectangles', {}).items():
                # 假设 add_rectangle 方法接受具体的维度作为参数
                defect.add_rectangle(rect_id, *rect_vals)  # 解包可能的矩形维度列表或元组

            frame_data.defects[defect_type] = defect

        all_frame_data[int(frame_id)] = frame_data

    return all_frame_data


def filter_anomalies_with_segmented_interpolation(values, segment_size=1000):
    filtered_values = np.copy(values)

    for start in range(0, len(values), segment_size):
        end = min(start + segment_size, len(values))
        segment_values = values[start:end]

        # 计算四分位数
        Q1 = np.percentile(segment_values, 25)
        Q3 = np.percentile(segment_values, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR - 0.1
        upper_bound = Q3 + 1.5 * IQR + 0.1

        anomalies = (segment_values > upper_bound) | (segment_values < lower_bound)
        normal_indices = np.where(~anomalies)[0]
        normal_values = segment_values[normal_indices]

        if normal_indices.size > 0:
            linear_interpolator = interp1d(normal_indices, normal_values,
                                           kind='linear', fill_value='extrapolate', bounds_error=False)
            filtered_values[start:end][anomalies] = linear_interpolator(np.where(anomalies)[0])
        else:
            filtered_values[start:end] = np.median(segment_values)

    return filtered_values

def plot_distances(frame_data_dict):
    # 提取每一帧的ID和对应的距离
    frame_ids = sorted(frame_data_dict.keys())
    distances = [frame_data_dict[frame_id].distance for frame_id in frame_ids if frame_data_dict[frame_id].distance is not None]

    # 应用异常值滤除
    # filtered_distances = filter_anomalies_with_segmented_interpolation(np.array(distances))

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(frame_ids, distances, marker='o', linestyle='-', color='b')
    plt.title('Distance per Frame')
    plt.xlabel('Frame ID')
    plt.ylabel('Distance (m)')
    plt.grid(True)
    plt.show()



class NumpyEncoder(json.JSONEncoder):
    """ 用于处理numpy数据类型的JSON编码器 """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 将numpy数组转换为列表
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def save_all_frame_data_to_json(all_frame_data, file_path):
    """ 将所有帧数据保存到JSON文件 """
    data_to_save = {}

    for frame_id, frame_data in all_frame_data.items():
        # 处理每个帧的数据
        defects_dict = {}
        for defect_type, defect in frame_data.defects.items():
            # 将缺陷的rectangles中的numpy数组转换为列表
            rectangles = {str(k): (v[0], v[1]) for k, v in defect.rectangles.items()}
            defects_dict[defect_type] = {
                "Count": defect.count,
                "Rectangles": rectangles
            }

        data_to_save[str(frame_id)] = {
            "Distance": frame_data.distance,
            "Defects": defects_dict
        }

    # 写入JSON文件
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data_to_save, file, cls=NumpyEncoder, indent=4, ensure_ascii=False)


def track_video_task(video_path, model_path, output_path, conf_thresholds):
    tracker = VideoTracker(video_path, model_path, output_path, conf_thresholds)
    tracker.track_video()
    return tracker.get_all_frame_data()


if __name__ == "__main__":
    video_path = "D:/workspaceTech/ultralytics/main/res/测试02片段/测试02片段.mp4"
    audio_path = "D:/workspaceTech/ultralytics/main/res/测试02片段/测试02片段.wav"

    video_paths = [video_path, video_path, video_path]

    model_path = 'D:/workspaceTech/ultralytics/weights/obb/17/best.pt'
    output_path = 'D:/workspaceTech/ultralytics/main/res/测试02片段/测试02片段_structure_tracked.mp4'
    conf_thresholds = {
            'QN': 0.4, 'YW': 0.6, 'GL': 0.8, 'FS': 0.8, 'CJ': 0.4, 'ZJ': 0.8, 'ZG': 0.4,
            'LGL': 0.4, 'MGL': 0.4, 'SGL': 0.4, 'SSGL': 0.4, 'LFS': 0.4, 'MFS': 0.4,
            'SFS': 0.4, 'SSFS': 0.4, 'SZ': 0.4, 'FM': 0.4, 'TL': 0.4, 'WZ': 0.4,
            'PS': 0.4, 'ZC': 0.8
        }

    model_paths = ['D:/workspaceTech/ultralytics/weights/obb/17/best.pt']
    output_paths = ['D:/workspaceTech/ultralytics/main/res/测试02片段/测试02片段_structure_tracked.mp4']

    conf_thresholds_list = [
        {
            'QN': 0.4, 'YW': 0.6, 'GL': 0.8, 'FS': 0.8, 'CJ': 0.4, 'ZJ': 0.8, 'ZG': 0.4,
            'LGL': 0.4, 'MGL': 0.4, 'SGL': 0.4, 'SSGL': 0.4, 'LFS': 0.4, 'MFS': 0.4,
            'SFS': 0.4, 'SSFS': 0.4, 'SZ': 0.4, 'FM': 0.4, 'TL': 0.4, 'WZ': 0.4,
            'PS': 0.4, 'ZC': 0.8
        }
    ]

    # 执行视频跟踪
    tracker = VideoTracker(video_path, model_path, output_path, conf_thresholds)
    tracker.track_video()
    all_frame_data = tracker.get_all_frame_data()

    # 打印跟踪数据
    for frame_id, frame_data in all_frame_data.items():
        print(f"Frame ID: {frame_id}")
        print(f"Distance: {frame_data.distance}")
        print("Defects:")
        for defect_type, defect in frame_data.defects.items():
            print(f"  {defect_type} - Count: {defect.count}, Rectangles: {defect.rectangles}")

    # 音频分析
    analyzer = AudioAnalyzer(audio_path, video_path)
    audio_json_path = "D:/workspaceTech/ultralytics/main/res/测试02片段/测试02片段_audio_frequencies.json"
    analyzer.save_audio_frequencies_to_json(audio_json_path)
    output_filtered_json = "D:/workspaceTech/ultralytics/main/res/测试02片段/测试02片段_audio_frequencies_filtered.json"
    analyzer.filter_and_merge_audio_frequencies(audio_json_path, output_filtered_json)

    # 从 JSON 文件读取并整合音频结果
    with open(output_filtered_json, 'r', encoding='utf-8') as file:
        audio_results = json.load(file)
        analyzer.integrate_audio_to_all_frame_data(all_frame_data, audio_results)

    ocr_before_json = "D:/workspaceTech/ultralytics/main/res/测试02片段/测试02片段ocr_before.json"
    save_all_frame_data_to_json(all_frame_data, ocr_before_json)

    # OCR 位置预测
    ocr_model_dir = "D:/workspaceTech/ultralytics/OCR/models/recognition/ch_PP-OCRv4_rec_infer"
    frame_ocr_processor = FrameOCR(video_path, ocr_model_dir)
    all_frame_data = frame_ocr_processor.process_video(all_frame_data)  # 更新已有数据

    ocr_after_json = "D:/workspaceTech/ultralytics/main/res/测试02片段/测试02片段ocr_after_json.json"
    save_all_frame_data_to_json(all_frame_data, ocr_after_json)

    # 从 JSON 文件中加载数据
    loaded_frame_data = load_frame_data_from_json(ocr_after_json)

    # 假设 loaded_frame_data 是从 JSON 加载的帧数据
    plot_distances(loaded_frame_data)

    # 计算总长度：找出第一帧和最后一帧的距离
    all_distances = [frame_data.distance for frame_data in all_frame_data.values() if frame_data.distance is not None]
    if all_distances:
        total_length = max(all_distances) - min(all_distances)
    else:
        total_length = 1

    ocr_after_json = "D:/workspaceTech/ultralytics/main/res/测试02片段/测试02片段ocr_after_json.json"
    save_all_frame_data_to_json(all_frame_data, ocr_after_json)

    # 缺陷信息处理
    video_width, video_height = 1920, 1080  # 示例分辨率，您可以根据视频元数据获取真实值
    defect_info_processor = DefectInfoProcessor(video_width, video_height)

    defect_info_processor.filter_frames(all_frame_data)
    defect_info_processor.process_all_frame_data(all_frame_data)
    defect_infos = defect_info_processor.get_defect_infos()

    print(total_length)

    Pipeline_json_path = "D:/workspaceTech/ultralytics/main/res/测试02片段/测试02片段_pipeline_defect_summary.json"
    PipelineMerge_json_path = "D:/workspaceTech/ultralytics/main/res/测试02片段/测试02片段_pipeline_Mergedefect_summary.json"
    Pipeline_csv_path = "D:/workspaceTech/ultralytics/main/res/测试02片段/测试02片段_pipeline_defect_summary.csv"
    evaluator = PipelineDefectEvaluator(defect_infos, total_length)  # 假设管道总长为1000米
    evaluator.save_evaluation_to_json(Pipeline_json_path)
    evaluator.merge_selective_defects()
    evaluator.save_evaluation_to_json(PipelineMerge_json_path)
    evaluator.save_evaluation_to_csv(Pipeline_csv_path)
    evaluator.print_defect_info()
    evaluator.print_summary()

