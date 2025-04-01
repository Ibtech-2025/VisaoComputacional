from ultralytics import YOLO

model = YOLO("yolov5s.pt")

results = model.predict("./image.png", save=True, conf=0.1)
