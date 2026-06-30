from ultralytics import YOLO

weight_path = "../../data/output/chess-model-yolov8m.pt"
chess_detector = YOLO(weight_path)

img_path = "../../data/output/images/board_screenshot.png"
results = chess_detector(img_path, conf=0.25)

frame_result = results[0]
print(frame_result.boxes.xywh)
print(frame_result.boxes.cls)