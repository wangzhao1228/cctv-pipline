import json

import concurrent.futures
from defect_infoSegmentold import DefectInfoProcessor, DamageScorer, PipelineDefectEvaluator,AudioAnalyzer
from frame_ocr import FrameOCR
from video_track_obb import VideoTracker
from video_track_obb import FrameDataMerger


def track_video_task(video_path, model_path, output_path):
    tracker = VideoTracker(video_path, model_path, output_path)
    tracker.track_video()
    return tracker.get_all_frame_data()

if __name__ == "__main__":
    video_path = "D:/workspaceTech/ultralytics/main/res/测试/测试2视频.mp4"
    audio_path = "D:/workspaceTech/ultralytics/main/res/测试/测试2音频.wav"


    video_path = "D:/workspaceTech/ultralytics/main/res/测试/测试2视频.mp4"

    # # OCR 位置预测
    # ocr_model_dir = "D:/workspaceTech/ultralytics/OCR/models/recognition/ch_PP-OCRv4_rec_infer"
    # frame_ocr_processor = FrameOCR(video_path, ocr_model_dir)
    # frame_ocr_processor.process_video(all_frame_data)  # 更新已有数据

    # 计算总长度：找出第一帧和最后一帧的距离
    # all_distances = [frame_data.distance for frame_data in all_frame_data.values() if frame_data.distance is not None]
    # if all_distances:
    #     total_length = max(all_distances) - min(all_distances)
    # else:
    #     total_length = 1

    # 音频分析
    analyzer = AudioAnalyzer(audio_path, video_path)
    audio_json_path = "D:/workspaceTech/ultralytics/main/res/测试/测试2视频audio_frequencies.json"
    analyzer.save_audio_frequencies_to_json(audio_json_path)
    # analyzer.plot_frequency_and_power_distributions(audio_json_path)
    # analyzer.plot_threshold_effects(audio_json_path)
    analyzer.plot_combined_thresholds(audio_json_path)
    output_filtered_json = "D:/workspaceTech/ultralytics/main/res/测试/测试2视频audio_frequencies_filtered.json"
    analyzer.filter_and_merge_audio_frequencies(audio_json_path, output_filtered_json)

