import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
# model = YOLO('yolov8n.pt')
# model = YOLO('yolov8n-obb.yaml').load('yolov8n-obb.pt')
model = YOLO('D:/workspaceTech/ultralytics/weights/obb/obb_best.pt')

# Open the video file
video_path = "D:/workspaceTech/ultralytics/tests/huaxiangroad/4.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4
out_path = "D:/workspaceTech/ultralytics/runs/obb/track/4_tracked.mp4"
out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the frame into the file 'output.avi'
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()