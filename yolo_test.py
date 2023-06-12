import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/home/hiepnt/Projects/yolov8/runs/detect/yolov8n_pti_quality2/weights_test/best.pt')

# results = self.model.predict(img, stream=False)                 # run prediction on img
# for result in results:                                         # iterate results
#     boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
#     for box in boxes:                                          # iterate boxes
#         r = box.xyxy[0].astype(int)                            # get corner points as int
#         print(r)                                               # print boxes
#         cv2.rectangle(img, r[:2], r[2:], (255, 255, 255), 2)   # draw boxes on img
img_path = '/opt/github/pytorch-to-onnx/image_1686301058.png'

model.predict(img_path, save=True, imgsz=640, conf=0.5)





