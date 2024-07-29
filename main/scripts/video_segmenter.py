import cv2

class VideoSegmenter:
    def __init__(self, video_path, num_segments=10):
        self.video_path = video_path
        self.num_segments = num_segments
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.segment_length = self.total_frames // num_segments
        self.cap.release()

    def get_segments(self):
        """生成每个段的起始和结束帧索引"""
        segments = [(i * self.segment_length, min((i + 1) * self.segment_length, self.total_frames)) for i in range(self.num_segments)]
        if segments[-1][1] != self.total_frames:  # 确保最后一段覆盖到视频末尾
            segments[-1] = (segments[-1][0], self.total_frames)
        return segments
