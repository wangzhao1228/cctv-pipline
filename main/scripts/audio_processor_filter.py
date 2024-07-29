import librosa
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2

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

    def get_total_duration(self):
        """
        获取音频文件的总时长（秒）
        Returns:
        float: 音频文件的总时长
        """
        return len(self.waveform) / self.sr

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

    def plot_power_spectrum(self):
        """
        绘制功率谱
        """
        frames, frame_length, frame_num = self.compute_frames()
        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = mag_frames ** 2 / NFFT

        plt.figure()
        plt.imshow(20 * np.log10(pow_frames[40:].T), cmap=plt.cm.jet, aspect='auto')
        plt.yticks([0, 64, 128, 192, 256], np.array([0, 64, 128, 192, 256]) * self.sr / NFFT)
        plt.show()

    def print_frequencies_and_dbs(self, frame_size=0.025, frame_stride=0.01):
        """
        打印每个时刻的频率和分贝值
        Args:
        frame_size (float): 每一帧的长度，单位为秒
        frame_stride (float): 相邻两帧的间隔，单位为秒
        """
        frames, frame_length, frame_num = self.compute_frames(frame_size, frame_stride)
        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = mag_frames ** 2 / NFFT
        pow_frames = np.where(pow_frames == 0, np.finfo(float).eps, pow_frames)  # 替换零值
        freqs = np.linspace(0, self.sr / 2, NFFT // 2 + 1)

        time_indices = librosa.frames_to_time(range(frame_num), sr=self.sr, hop_length=int(frame_stride * self.sr))
        print("Time (s), Frequency (Hz), Decibel (dB)")
        for i, t in enumerate(time_indices):
            max_freq_idx = np.argmax(pow_frames[i][:len(freqs)])
            max_freq = freqs[max_freq_idx]
            max_db = 20 * np.log10(pow_frames[i][max_freq_idx])
            video_frame = self.audio_time_to_video_frame(t)
            print(f"Frame {video_frame}: Time {t:.3f}, Frequency {max_freq:.2f}, Decibel {max_db:.2f}")

    def save_audio_frequencies_to_json(self, json_path, frame_size=0.025, frame_stride=0.01):
        """
        将每一视频帧对应的音频频率和分贝值保存到JSON文件
        Args:
        json_path (str): JSON文件保存路径
        frame_size (float): 每一帧的长度，单位为秒
        frame_stride (float): 相邻两帧的间隔，单位为秒
        """
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

    def filter_and_merge_audio_frequencies(self, input_json_path, output_json_path, freq_threshold=4000.0, db_threshold=-110.0, duration_seconds=10.0, frame_window=30, max_gap_seconds=10):
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
                frame_count = 0  # Reset the frame count for each valid frame
            else:
                frame_count += 1

            # Check if the current segment is active and the frame window or time has exceeded
            if active_segment and (frame_count > frame_window or current_time - active_segment['start_time'] > duration_seconds):
                if current_time - active_segment['start_time'] >= duration_seconds:
                    # Close the active segment
                    active_segment['end_frame'] = data[i-1]['video_frame']
                    active_segment['end_time'] = data[i-1]['time']
                    results.append(active_segment)
                # Reset the segment
                active_segment = None
                frame_count = 0

        # Check if there's an active segment at the end of data
        if active_segment and current_time - active_segment['start_time'] >= duration_seconds:
            active_segment['end_frame'] = data[-1]['video_frame']
            active_segment['end_time'] = data[-1]['time']
            results.append(active_segment)

        # Merge consecutive segments with no time gap
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

def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = seconds % 60
    return f"{hours:02}:{minutes:02}:{sec:05.2f}"

# Example usage
audio_path = "D:/workspaceTech/ultralytics/main/res/漏点/武威路（真南路-雪松路）_DN500_20221115231140/武威路（真南路-雪松路）_DN500_20221115231140.wav"
video_path = "D:/workspaceTech/ultralytics/main/res/漏点/武威路（真南路-雪松路）_DN500_20221115231140.mp4"
# audio_path = "D:/WorkDocument/10_项目/一供/项目汇报/汇晟283/Video-20230307-235103.WAV"
# video_path = "D:/WorkDocument/10_项目/一供/项目汇报/汇晟283/Video-20230307-235103.avi"
analyzer = AudioAnalyzer(audio_path, video_path)
analyzer.plot_power_spectrum()
analyzer.print_frequencies_and_dbs()
json_path = "D:/workspaceTech/ultralytics/main/res/漏点/audio_frequencies.json"
# json_path = "D:/WorkDocument/10_项目/一供/项目汇报/汇晟283/audio_frequencies.json"
analyzer.save_audio_frequencies_to_json(json_path)
# output_filtered_json = "D:/WorkDocument/10_项目/一供/项目汇报/汇晟283/audio_frequencies_filtered.json"
output_filtered_json = "D:/workspaceTech/ultralytics/main/res/漏点/audio_frequencies_filtered.json"
analyzer.filter_and_merge_audio_frequencies(json_path, output_filtered_json, freq_threshold=4000.0, db_threshold=-110.0, duration_seconds=5.0, frame_window=200)

with open(output_filtered_json, 'r', encoding='utf-8') as file:
    data = json.load(file)
    for entry in data:
        print(entry)

with open(output_filtered_json, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Print the data with start_time and end_time converted to hours:minutes:seconds format
for entry in data:
    start_time_hms = seconds_to_hms(entry['start_time'])
    end_time_hms = seconds_to_hms(entry['end_time'])
    print({
        'start_frame': entry['start_frame'],
        'start_time': start_time_hms,
        'end_frame': entry['end_frame'],
        'end_time': end_time_hms
    })
