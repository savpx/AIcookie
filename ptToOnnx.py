from ultralytics import YOLO

# Экспортируем обученную модель в ONNX (для авторазметки в X-AnyLabeling)
model = YOLO("runs/detect/cookie_detector/weights/best.pt")
model.export(format="onnx", opset=12, simplify=True)
