import json
import numpy as np

import concurrent.futures
from defect_infoSegmentold import DefectInfoProcessor, DamageScorer, PipelineDefectEvaluator, AudioAnalyzer
from frame_ocr_interpolate import FrameOCR
from video_track_obb import VideoTracker
from video_track_obb import FrameDataMerger


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


def track_video_task(video_path, model_path, output_path):
    tracker = VideoTracker(video_path, model_path, output_path)
    tracker.track_video()
    return tracker.get_all_frame_data()


if __name__ == "__main__":
    video_path = "D:/workspaceTech/ultralytics/main/res/测试/测试片段1/测试片段1.mp4"
    audio_path = "D:/workspaceTech/ultralytics/main/res/测试/测试片段1/测试片段1.wav"

    video_paths = [video_path, video_path, video_path]

    model_paths = ['D:/workspaceTech/ultralytics/weights/obb/17/last.pt']
    output_paths = ['D:/workspaceTech/ultralytics/main/res/测试/测试片段1/测试片段1_tracked.mp4']

    # 创建FrameDataMerger实例
    merger = FrameDataMerger()

    # 使用线程池并行运行视频跟踪
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(track_video_task, vp, mp, op) for vp, mp, op in
                   zip(video_paths, model_paths, output_paths)]

        # 等待所有的future完成，并收集结果
        for future in concurrent.futures.as_completed(futures):
            try:
                frame_data_dict = future.result()
                # 合并到总的frame_data_dict中
                merger.merge_frame_data_dicts(frame_data_dict)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    # 获取合并后的所有帧数据
    all_frame_data = merger.get_merged_data()

    # 打印跟踪数据
    for frame_id, frame_data in all_frame_data.items():
        print(f"Frame ID: {frame_id}")
        print(f"Distance: {frame_data.distance}")
        print("Defects:")
        for defect_type, defect in frame_data.defects.items():
            print(f"  {defect_type} - Count: {defect.count}, Rectangles: {defect.rectangles}")

    # 音频分析
    analyzer = AudioAnalyzer(audio_path, video_path)
    audio_json_path = "D:/workspaceTech/ultralytics/main/res/测试/测试片段11/测试片段1_audio_frequencies.json"
    analyzer.save_audio_frequencies_to_json(audio_json_path)
    output_filtered_json = "D:/workspaceTech/ultralytics/main/res/测试/测试片段11/测试片段1_audio_frequencies_filtered.json"
    analyzer.filter_and_merge_audio_frequencies(audio_json_path, output_filtered_json)

    # 从 JSON 文件读取并整合音频结果
    with open(output_filtered_json, 'r', encoding='utf-8') as file:
        audio_results = json.load(file)
        analyzer.integrate_audio_to_all_frame_data(all_frame_data, audio_results)

    ocr_before_json = "D:/workspaceTech/ultralytics/main/res/测试/测试片段11/测试片段1_ocr_before.json"
    save_all_frame_data_to_json(all_frame_data, ocr_before_json)

    # OCR 位置预测
    ocr_model_dir = "D:/workspaceTech/ultralytics/OCR/models/recognition/ch_PP-OCRv4_rec_infer"
    frame_ocr_processor = FrameOCR(video_path, ocr_model_dir)
    frame_ocr_processor.process_video(all_frame_data)  # 更新已有数据

    # 计算总长度：找出第一帧和最后一帧的距离
    all_distances = [frame_data.distance for frame_data in all_frame_data.values() if frame_data.distance is not None]
    if all_distances:
        total_length = max(all_distances) - min(all_distances)
    else:
        total_length = 1

    ocr_after_json = "D:/workspaceTech/ultralytics/main/res/测试/测试片段11/测试片段1ocr_after_json.json"
    save_all_frame_data_to_json(all_frame_data, ocr_after_json)

    # 缺陷信息处理
    video_width, video_height = 1920, 1080  # 示例分辨率，您可以根据视频元数据获取真实值
    defect_info_processor = DefectInfoProcessor(video_width, video_height)

    defect_info_processor.filter_defects(all_frame_data)
    defect_info_processor.process_all_frame_data(all_frame_data)

    defect_infos = defect_info_processor.get_defect_infos()

    print(total_length)

    Pipeline_json_path = "D:/workspaceTech/ultralytics/main/res/测试/测试片段11/测试片段1_pipeline_defect_summary.json"
    PipelineMerge_json_path = "D:/workspaceTech/ultralytics/main/res/测试/测试片段11/测试片段1_pipeline_Mergedefect_summary.json"
    Pipeline_csv_path = "D:/workspaceTech/ultralytics/main/res/测试/测试片段11/测试片段1_pipeline_defect_summary.csv"
    evaluator = PipelineDefectEvaluator(defect_infos, total_length)  # 假设管道总长为1000米
    evaluator.save_evaluation_to_json(Pipeline_json_path)
    evaluator.merge_selective_defects()
    evaluator.save_evaluation_to_json(PipelineMerge_json_path)
    evaluator.save_evaluation_to_csv(Pipeline_csv_path)
    evaluator.print_defect_info()
    evaluator.print_summary()
