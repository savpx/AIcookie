import os
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

# Порог уверенности для каждого типа объекта
CONF = 0.5           # общий порог
CONF_GAME = 0.25     # печенька: кликаем даже при невысокой уверенности
CONF_CLOSE = 0.7     # рекламу закрываем только при высокой уверенности

# Раз в сколько секунд покупать улучшения (снизу вверх)
UPGRADE_INTERVAL = 10.0

# Кулдауны, чтобы одно и то же срабатывание не блокировало клики по печеньке.
# Если объект не исчезает после клика — это ложное срабатывание.
CLOSE_COOLDOWN = 1.5
BONUS_COOLDOWN = 1.0

DEBUG = os.environ.get('COOKIE_DEBUG') == '1'

# Имена классов улучшений
UPGRADE_CLASSES = [f"upgrade{i}" for i in range(1, 10)]


def click_at(center_x, center_y):
    pyautogui.click(center_x, center_y)


def detect(screenshot_np):
    """Возвращает самый уверенный бокс (x1,y1,x2,y2) для каждого класса."""
    results = model(screenshot_np, verbose=False)
    found = {}
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            name = model.names[int(box.cls[0])]
            if name not in found or confidence > found[name][0]:
                found[name] = (confidence, [float(v) for v in box.xyxy[0]])
    return {name: data for name, data in found.items()}


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def buy_upgrades(found):
    """Кликает по центру каждого улучшения снизу вверх (по возрастанию Y)."""
    upgrades = [
        (name, data[1]) for name, data in found.items()
        if name in UPGRADE_CLASSES and data[0] > CONF
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
    last_close_time = 0.0
    last_bonus_time = 0.0
    last_game_box = None      # запомненная позиция печеньки
    last_game_time = 0.0

    while True:
        # 1. Смотрим на экран
        screenshot = pyautogui.screenshot()
        screenshot_np = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        found = detect(screenshot_np)

        if DEBUG:
            summary = ", ".join(
                f"{n}={d[0]:.2f}" for n, d in sorted(found.items()))
            print(f"[detect] {summary or 'ничего'}")

        now = time.time()

        # 2. Закрыть рекламу (только уверенно и не чаще кулдауна)
        close = found.get('close')
        if close and close[0] > CONF_CLOSE and now - last_close_time > CLOSE_COOLDOWN:
            cx, cy = box_center(close[1])
            click_at(cx, cy)
            last_close_time = now
            print(f"Закрытие рекламы: ({cx:.0f}, {cy:.0f})")
            time.sleep(0.2)
            continue

        # 3. Собрать бонусную печеньку
        bonus = found.get('bonus')
        if bonus and bonus[0] > CONF and now - last_bonus_time > BONUS_COOLDOWN:
            cx, cy = box_center(bonus[1])
            click_at(cx, cy)
            last_bonus_time = now
            print(f"Бонус собран: ({cx:.0f}, {cy:.0f})")
            time.sleep(0.1)
            continue

        # 4. Каждые UPGRADE_INTERVAL секунд покупаем улучшения снизу вверх
        if now - last_upgrade_time >= UPGRADE_INTERVAL:
            if buy_upgrades(found):
                last_upgrade_time = now
                continue

        # 5. Основное действие — быстро кликаем по игровому полю (печеньке)
        game = found.get('game')
        if game and game[0] > CONF_GAME:
            # свежая позиция печеньки — запоминаем
            last_game_box = game[1]
            last_game_time = now
        elif last_game_box is None or now - last_game_time > 5.0:
            # давно не видели печеньку — ждём, пока модель её найдёт
            game = None

        if last_game_box is not None and now - last_game_time <= 5.0:
            cx, cy = box_center(last_game_box)
            for _ in range(CLICKS_PER_BURST):
                pyautogui.click(cx, cy)
                time.sleep(CLICK_DELAY)
            continue

        # Ничего не нашли — короткая пауза и снова
        time.sleep(0.05)


if __name__ == '__main__':
    print("Запуск. Для остановки уведите мышь в левый верхний угол.")
    if DEBUG:
        print("Режим отладки: каждый цикл печатает найденные объекты.")
    main_loop()
