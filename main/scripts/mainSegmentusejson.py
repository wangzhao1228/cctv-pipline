import json
import matplotlib.pyplot as plt
import numpy as np

import concurrent.futures

from scipy.interpolate import interp1d

from defect_infoSegment import DefectInfoProcessor, DamageScorer, PipelineDefectEvaluator, AudioAnalyzer, FrameData, \
    Defect


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
        Q1 = np.percentile(segment_values, 45)
        Q3 = np.percentile(segment_values, 55)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR -1
        upper_bound = Q3 + 1.5 * IQR +1

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


if __name__ == "__main__":
    ocr_after_json = "D:/workspaceTech/ultralytics/main/res/testload/测试2视频last.json"

    # 从 JSON 文件中加载数据
    loaded_frame_data = load_frame_data_from_json(ocr_after_json)
    # 假设 loaded_frame_data 是从 JSON 加载的帧数据
    plot_distances(loaded_frame_data)

    # # 缺陷信息处理
    # video_width, video_height = 1920, 1080  # 示例分辨率，您可以根据视频元数据获取真实值
    # defect_info_processor = DefectInfoProcessor(video_width, video_height)
    #
    # defect_info_processor.filter_frames(loaded_frame_data)
    # defect_info_processor.process_all_frame_data(loaded_frame_data)
    #
    # defect_infos = defect_info_processor.get_defect_infos()
    # total_length = defect_info_processor.get_pipeline_length()
    # print("Total length from loaded data:", total_length)
    #
    # Pipeline_json_path = "D:/workspaceTech/ultralytics/main/res/testload/测试2ocr_after_json070901_pipeline_defect_summary.json"
    # PipelineMerge_json_path = "D:/workspaceTech/ultralytics/main/res/testload/测试2ocr_after_json070901_pipeline_Mergedefect_summary.json"
    # Pipeline_csv_path = "D:/workspaceTech/ultralytics/main/res/testload/测试2ocr_after_json070901_pipeline_defect_summary.csv"
    # PipelineMerge_csv_path = "D:/workspaceTech/ultralytics/main/res/testload/测试2ocr_after_json070901_pipeline_PipelineMerge_summary.csv"
    #
    # evaluator = PipelineDefectEvaluator(defect_infos, total_length)
    # evaluator.save_evaluation_to_json(Pipeline_json_path)
    # evaluator.save_evaluation_to_csv(Pipeline_csv_path)
    # evaluator.merge_selective_defects()
    # evaluator.save_evaluation_to_json(PipelineMerge_json_path)
    # evaluator.save_evaluation_to_csv(PipelineMerge_csv_path)
    # evaluator.print_defect_info()
    # evaluator.print_summary()


