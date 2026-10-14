import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Загрузка обученной модели
model = YOLO('runs/detect/cookie_detector/weights/best.pt')

# Failsafe: резко уведите мышь в левый верхний угол — программа остановится
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# Сколько кликов по печеньке делать подряд, прежде чем снова "посмотреть" на экран
CLICKS_PER_BURST = 30
CLICK_DELAY = 0.01  # пауза между кликами (0.01 = ~100 кликов/сек)

CONF = 0.7  # порог уверенности


def click_at(center_x, center_y):
    pyautogui.click(center_x, center_y)


def detect(screenshot_np):
    """Возвращает самый уверенный бокс (x1,y1,x2,y2) для каждого класса."""
    results = model(screenshot_np)
    found = {}
    for result in results:
        for box in result.boxes:
            confidence = box.conf[0]
            if confidence <= CONF:
                continue
            name = model.names[int(box.cls[0])]
            if name not in found:
                found[name] = [float(v) for v in box.xyxy[0]]
    return found


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def main_loop():
    while True:
        # 1. Смотрим на экран
        screenshot = pyautogui.screenshot()
        screenshot_np = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        found = detect(screenshot_np)

        # 2. Приоритет: закрыть рекламу
        if 'close' in found:
            cx, cy = box_center(found['close'])
            click_at(cx, cy)
            print(f"Закрытие рекламы: ({cx:.0f}, {cy:.0f})")
            time.sleep(0.3)
            continue

        # 3. Собрать бонусную печеньку
        if 'bonus' in found:
            cx, cy = box_center(found['bonus'])
            click_at(cx, cy)
            print(f"Бонус собран: ({cx:.0f}, {cy:.0f})")
            time.sleep(0.1)
            continue

        # 4. Купить улучшения: кликаем по строкам панели сверху вниз
        #    (верхние улучшения самые дешёвые и обычно доступны)
        if 'upgrade' in found:
            x1, y1, x2, y2 = found['upgrade']
            cx = (x1 + x2) / 2
            h = y2 - y1
            # кликаем по строкам панели сверху вниз
            # (верхние улучшения самые дешёвые и обычно доступны)
            for frac in (0.15, 0.30, 0.45, 0.60):
                cy = y1 + h * frac
                pyautogui.click(cx, cy)
                time.sleep(0.05)
            print(f"Покупка улучшений в панели x={cx:.0f}")
            time.sleep(0.1)
            continue

        # 5. Иначе — быстро и непрерывно кликаем по игровому полю
        if 'game' in found:
            cx, cy = box_center(found['game'])
            for _ in range(CLICKS_PER_BURST):
                pyautogui.click(cx, cy)
                time.sleep(CLICK_DELAY)
            continue

        # Ничего не нашли — короткая пауза и снова
        time.sleep(0.1)


if __name__ == '__main__':
    print("Запуск. Для остановки уведите мышь в левый верхний угол.")
    main_loop()
