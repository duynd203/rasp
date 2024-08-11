import cv2
from ultralytics import YOLO
import time
import numpy as np



# def preprocessing(image):
#     height, width = image.shape[:2]
#     range_w, range_h = 200,0
#     # range_w =  200 #150
#     # range_h =  40 #150
#     return image[range_h : height - range_h, range_w : width - range_w]


# def preprocessing(image):
#     brightness = 30
#     image = np.where((255 - image) < brightness, 255, image + brightness)
#     # Tăng độ tương phản
#     # contrast = 1.2
#     # image = np.clip(contrast * image, 0, 255).astype(np.uint8)
#     return image
    

onnx_model = YOLO("yolov10s320_267_10class.onnx")

# Open the video file
cap = cv2.VideoCapture(0)
# Thiết lập kích thước khung hình
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'RGB3'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440 )
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720  )
# Initialize variables for FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    if success:
        # Measure the start time
        start_tick = cv2.getTickCount()
        results = onnx_model(frame,  conf = 0.4, iou = 0.4, imgsz=320)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Calculate FPS
        frame_count += 1
        end_tick = cv2.getTickCount()
        time_spent = (end_tick - start_tick) / cv2.getTickFrequency()
        fps = 1 / time_spent
        # print(fps)
        # Display the FPS on the frame
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
       
        
    else:
        # Break the loop if the end of the video is reached
        break

# Calculate average FPS
end_time = time.time()
total_time = end_time - start_time
average_fps = frame_count / total_time
print(f"Average FPS: {average_fps:.2f}")

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
