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
    # video_path = 'D:/workspaceTech/ultralytics/main/res/4.mp4'
    # video_path = 'D:/workspaceTech/ultralytics/main/res/0509.mp4'
    # video_path = "D:/workspaceTech/ultralytics/main/res/漏点/武威路（真南路-雪松路）_DN500_20221115231140.mp4"
    # audio_path = "D:/workspaceTech/ultralytics/main/res/漏点/武威路（真南路-雪松路）_DN500_20221115231140/武威路（真南路-雪松路）_DN500_20221115231140.wav"
    video_path = "D:/workspaceTech/ultralytics/main/res/ppt/中度腐蚀.mp4"
    # audio_path = "D:/workspaceTech/ultralytics/main/res/ppt/重度.WAV"

    # video_path = "D:/workspaceTech/ultralytics/main/res/0525/0525.mp4"

    video_paths = [video_path, video_path, video_path]

    model_paths = ['D:/workspaceTech/ultralytics/weights/obb/GLFS/best.pt',
                   'D:/workspaceTech/ultralytics/weights/obb/QNCJ/best.pt',
                   'D:/workspaceTech/ultralytics/weights/obb/ZGYW/best.pt',]
    output_paths = ['D:/workspaceTech/ultralytics/main/res/ppt/中度腐蚀_GLFS_tracked.mp4',
                    'D:/workspaceTech/ultralytics/main/res/ppt/中度腐蚀_QNCJ_tracked.mp4',
                    'D:/workspaceTech/ultralytics/main/res/ppt/中度腐蚀_ZGYW_tracked.mp4']

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
