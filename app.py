import time

import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO

# Загрузка обученной модели
model = YOLO('runs/detect/cookie_detector/weights/best.pt')

# Failsafe: резко уведите мышь в левый верхний угол — программа остановится
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# Сколько кликов по печеньке делать подряд, прежде чем снова "посмотреть" на экран
CLICKS_PER_BURST = 30
CLICK_DELAY = 0.01  # пауза между кликами (0.01 = ~100 кликов/сек)

CONF = 0.5  # порог уверенности

# Раз в сколько секунд покупать улучшения (снизу вверх)
UPGRADE_INTERVAL = 10.0

# Имена классов улучшений в порядке снизу вверх задаётся координатой Y,
# а не номером, поэтому просто собираем все, что начинаются с "upgrade"
UPGRADE_CLASSES = [f"upgrade{i}" for i in range(1, 10)]


def click_at(center_x, center_y):
    pyautogui.click(center_x, center_y)


def detect(screenshot_np):
    """Возвращает самый уверенный бокс (x1,y1,x2,y2) для каждого класса."""
    results = model(screenshot_np, verbose=False)
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


def buy_upgrades(found):
    """Кликает по центру каждого улучшения снизу вверх (по возрастанию Y)."""
    upgrades = [
        (name, box) for name, box in found.items()
        if name in UPGRADE_CLASSES
    ]
    if not upgrades:
        return False
    # снизу вверх: самый нижний (больший Y) кликается первым
    upgrades.sort(key=lambda item: box_center(item[1])[1], reverse=True)
    for name, box in upgrades:
        cx, cy = box_center(box)
        click_at(cx, cy)
        print(f"Покупка {name}: ({cx:.0f}, {cy:.0f})")
        time.sleep(0.3)
    return True


def main_loop():
    last_upgrade_time = 0.0
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

        # 4. Каждые UPGRADE_INTERVAL секунд покупаем улучшения снизу вверх
        now = time.time()
        if now - last_upgrade_time >= UPGRADE_INTERVAL:
            if buy_upgrades(found):
                last_upgrade_time = now
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
