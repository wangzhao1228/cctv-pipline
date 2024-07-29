import cv2
import pytesseract
import numpy as np

# 指定 Tesseract-OCR 的安装路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(frame):
    # 转换为灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊，去除噪点
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 二值化
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    return thresh

# 打开视频
cap = cv2.VideoCapture('D:/workspaceTech/ultralytics/OCR/4.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = preprocess_image(frame)

    # 使用 OCR 识别整个帧的文本
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_frame, lang='chi_sim')

    # 剔除空格
    text = text.replace(" ", "")

    # 打印识别的文本
    print(text)

    # 显示识别的文本（可选）
    # cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)

    # 显示带有文本的视频帧
    # cv2.imshow('Video with Text', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
