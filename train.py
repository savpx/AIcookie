from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Загрузка предобученной модели
    model = YOLO('yolov8n.pt')

    # 2. Запуск обучения
    results = model.train(
        data='data.yaml',
        epochs=100,          # количество эпох
        imgsz=640,           # размер изображения
        batch=8,
        workers=0,           # отключаем многопоточность (проблемы на Windows)
        name='cookie_detector',
        exist_ok=True,       # перезапуск в ту же папку
        patience=50          # остановка, если нет улучшений 50 эпох подряд
    )
