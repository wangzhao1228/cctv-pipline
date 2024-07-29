import cv2
from ultralytics import YOLO
from defect_infoSegment import FrameData
class VideoTracker:
    def __init__(self, video_path, model_path, output_path, conf_thresholds):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.output_path = output_path
        self.conf_thresholds = conf_thresholds
        self.frame_data_dict = {}  # 使用字典存储帧数据

    def track_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))
        frame_id = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_id not in self.frame_data_dict:
                self.frame_data_dict[frame_id] = FrameData(frame_id)
            frame_data = self.frame_data_dict[frame_id]

            results = self.model.track(frame, persist=True, conf=0.2)
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
                        conf_threshold = self.conf_thresholds.get(defect_name, 0.35)
                        if conf >= conf_threshold:
                            frame_data.add_defect(defect_name, defect_id, width, height, x_center, y_center, rot, conf)

                if hasattr(result, 'plot'):
                    annotated_frame = result.plot()  # 对每个结果调用plot
                    out.write(annotated_frame)  # 将标注后的帧写入输出文件
                    # cv2.imshow("YOLOv8 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_id += 1

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

class FrameDataMerger:
    def __init__(self):
        # 初始化空字典以存储合并的FrameData
        self.merged_frame_data = {}

    def merge_frame_data_dicts(self, *frame_data_dicts):
        """合并多个frame_data_dict到一个字典中。

        Args:
            *frame_data_dicts: 一个或多个frame_data_dict字典，每个都包含FrameData实例。
        """
        for frame_data_dict in frame_data_dicts:
            for frame_id, frame_data in frame_data_dict.items():
                if frame_id not in self.merged_frame_data:
                    self.merged_frame_data[frame_id] = FrameData(frame_id)

                # 合并当前frame_data的缺陷到总的frame_data中
                for defect_type, defect in frame_data.defects.items():
                    for defect_id, dimensions in defect.rectangles.items():
                        self.merged_frame_data[frame_id].add_defect(
                            defect_type, defect_id, *dimensions
                        )

    def get_merged_data(self):
        """返回合并后的FrameData字典"""
        return self.merged_frame_data
