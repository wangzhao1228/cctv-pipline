import json

from defect_info import DefectInfoProcessor, DamageScorer
from frame_ocr import FrameOCR
from video_track_obb import VideoTracker
from defect_info import AudioAnalyzer

if __name__ == "__main__":
    # video_path = 'D:/workspaceTech/ultralytics/main/res/4.mp4'
    # video_path = 'D:/workspaceTech/ultralytics/main/res/0509.mp4'
    # video_path = "D:/workspaceTech/ultralytics/main/res/漏点/武威路（真南路-雪松路）_DN500_20221115231140.mp4"
    # audio_path = "D:/workspaceTech/ultralytics/main/res/漏点/武威路（真南路-雪松路）_DN500_20221115231140/武威路（真南路-雪松路）_DN500_20221115231140.wav"
    video_path = "D:/WorkDocument/10_项目/一供/项目汇报/金沙江路1161/金沙江路340_370.mp4"
    audio_path = "D:/WorkDocument/10_项目/一供/项目汇报/金沙江路1161/金沙江路340_370.WAV"

    # OBB 追踪
    model_path = 'D:/workspaceTech/ultralytics/weights/obb/obb_best.pt'
    # output_path = 'D:/workspaceTech/ultralytics/runs/obb/track/武威路（真南路-雪松路）_DN500_20221115231140_tracked.mp4'
    output_path = 'D:/WorkDocument/10_项目/一供/项目汇报/金沙江路1161/金沙江路_tracked.mp4'

    tracker = VideoTracker(video_path, model_path, output_path)
    tracker.track_video()
    all_frame_data = tracker.get_all_frame_data()

    # 打印跟踪数据
    for frame_id, frame_data in all_frame_data.items():
        print(f"Frame ID: {frame_id}")
        print(f"Distance: {frame_data.distance}")
        print("Defects:")
        for defect_type, defect in frame_data.defects.items():
            print(f"  {defect_type} - Count: {defect.count}, Rectangles: {defect.rectangles}")

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

    # 音频分析
    analyzer = AudioAnalyzer(audio_path, video_path)
    audio_json_path = "D:/WorkDocument/10_项目/一供/项目汇报/金沙江路1161/audio_frequencies.json"
    analyzer.save_audio_frequencies_to_json(audio_json_path)
    output_filtered_json = "D:/WorkDocument/10_项目/一供/项目汇报/金沙江路1161/audio_frequencies_filtered.json"
    analyzer.filter_and_merge_audio_frequencies(audio_json_path, output_filtered_json)

    # 从 JSON 文件读取并整合音频结果
    with open(output_filtered_json, 'r', encoding='utf-8') as file:
        audio_results = json.load(file)
        analyzer.integrate_audio_to_all_frame_data(all_frame_data, audio_results)

    # 缺陷信息处理
    video_width, video_height = 1920, 1080  # 示例分辨率，您可以根据视频元数据获取真实值
    defect_info_processor = DefectInfoProcessor(video_width, video_height)

    # 假设 all_frame_data 是一个字典
    print("First 10 elements of all_frame_data:")

    # 使用 items() 获取所有键值对，并使用 list() 转换后进行切片
    for i, (frame_id, frame_data) in enumerate(list(all_frame_data.items())[:10], start=1):
        print(f"Element {i}: Frame ID = {frame_id}")
        print(f"  Frame ID in object: {frame_data.frame_id}")
        print(f"  Distance: {frame_data.distance}")

        # 打印每种缺陷类型的详细信息
        for defect_type, defect in frame_data.defects.items():
            print(f"  Defect Type: {defect_type}")
            print(f"    Count: {defect.count}")
            print(f"    Rectangles: {defect.rectangles}")
    defect_info_processor.process_all_frame_data(all_frame_data)

    # 打印缺陷信息
    defect_infos = defect_info_processor.get_defect_infos()
    for defect_info in defect_infos:
        print(defect_info)

    print("总长度为：")
    print(total_length)
    scorer = DamageScorer(defect_infos, total_length)
    scorer.print_summary()
    total_score = scorer.calculate_score()
    print(f"受损状态评分: {total_score}")

    # 将 `defect_infos` 和 `total_score` 写入 JSON 文件
    summary_json_path = "D:/WorkDocument/10_项目/一供/项目汇报/金沙江路1161/defect_summary.json"
    summary_data = {
        "defect_infos": defect_infos,
        "total_score": total_score
    }

    # 打印 OCR 数据
    for frame_id, frame_data in all_frame_data.items():
        print(f"Frame ID: {frame_id}")
        print(f"Distance: {frame_data.distance}")
        print("Defects:")
        for defect_type, defect in frame_data.defects.items():
            print(f"  {defect_type} - Count: {defect.count}, Rectangles: {defect.rectangles}")
