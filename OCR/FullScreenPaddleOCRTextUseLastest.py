import cv2
from paddleocr import PaddleOCR
from collections import deque
from statistics import mode
import numpy as np

# 创建 OCR 对象
ocr = PaddleOCR(rec_model_dir=r"D:\workspaceTech\ultralytics\OCR\models\recognition\ch_PP-OCRv4_rec_infer",use_angle_cls=True, lang='ch')
# ocr = PaddleOCR(rec_model_dir=r"D:\workspaceTech\ultralytics\OCR\models\detection\b4",use_angle_cls=True, lang='ch')

# 打开视频文件
cap = cv2.VideoCapture('D:/workspaceTech/ultralytics/OCR/4.mp4')
frame_buffer = deque(maxlen=10)  # 存储最近5帧的识别结果


def process_frame(frame):
    """预处理图像并进行OCR识别"""
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ocr_results = ocr.ocr(processed_frame, cls=True)
    texts = []
    for line in ocr_results:
        if line is not None:
            texts.extend([line[i][1][0] for i in range(len(line))])
        else:
            print("该帧图片没有文本检测目标")
    return texts

def merge_similar_texts(texts, threshold=0.6):
    """合并相似的文本"""
    merged_texts = []
    for text in texts:
        if not merged_texts:
            merged_texts.append(text)
        else:
            similarity = similar_text(merged_texts[-1], text)
            if similarity >= threshold:
                merged_texts[-1] += " " + text
            else:
                merged_texts.append(text)
    return merged_texts

def edit_distance(text1, text2):
    """计算两个文本的编辑距离"""
    # 初始化编辑距离矩阵
    m, n = len(text1), len(text2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    # 计算编辑距离
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def similar_text(text1, text2):
    """计算两个文本的相似度（使用编辑距离）"""
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 0.0
    distance = edit_distance(text1, text2)
    similarity = 1 - distance / max_len
    return similarity

def post_processing(text):

        return text

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_texts = process_frame(frame)
    print(current_texts)
    frame_buffer.append(current_texts)

    if len(frame_buffer) == 10:
        # 检查缓存中所有帧的识别结果
        all_texts = [text for frame_texts in frame_buffer for text in frame_texts]
        # 合并相似的文本
        merged_texts = merge_similar_texts(all_texts)
        # 使用众数来确定最常见的文本
        try:
            most_common_text = mode(merged_texts)
            print("Most Common Text (Before Post-Processing): ", most_common_text)
            # 后处理，根据实际情况对识别结果进行校准
            # most_common_text = post_processing(most_common_text)
            # print("Most Common Text (After Post-Processing): ", most_common_text)
        except:
            print("No consensus found")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

