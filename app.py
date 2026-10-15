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

# Сколько кликов по печеньке делать за один цикл (до нового "взгляда" на экран)
CLICKS_PER_BURST = 20
CLICK_DELAY = 0.008  # пауза между кликами

# Пороги уверенности для каждого типа объекта
CONF_GAME = 0.15     # печенька: кликаем даже при невысокой уверенности
CONF_BONUS = 0.5     # бонус
CONF_CLOSE = 0.7     # рекламу закрываем только при высокой уверенности
CONF_UPGRADE = 0.4   # улучшения

# Порог детектора на уровне модели (должен быть НЕ выше самого низкого
# из порогов выше, иначе модель отбросит объекты раньше нашей проверки)
MODEL_CONF = 0.10

# Раз в сколько секунд покупать улучшения (снизу вверх)
UPGRADE_INTERVAL = 10.0

# Кулдауны, чтобы ложное срабатывание не спамило кликами
CLOSE_COOLDOWN = 2.0
BONUS_COOLDOWN = 1.5

# Сколько секунд помнить позицию печеньки, если её временно не видно
GAME_MEMORY = 8.0

DEBUG = os.environ.get('COOKIE_DEBUG') == '1'

# Имена классов улучшений
UPGRADE_CLASSES = [f"upgrade{i}" for i in range(1, 10)]

# Отладочные кадры
DEBUG_DIR = "debug_frames"
DEBUG_FRAME_LIMIT = 40
_debug_saved = 0


def click_at(center_x, center_y):
    pyautogui.click(center_x, center_y)


def detect(screenshot_np):
    """Возвращает самый уверенный бокс (x1,y1,x2,y2) для каждого класса."""
    results = model(screenshot_np, conf=MODEL_CONF, verbose=False)
    found = {}
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            name = model.names[int(box.cls[0])]
            if name not in found or confidence > found[name][0]:
                found[name] = (confidence, [float(v) for v in box.xyxy[0]])
    return found


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def save_debug_frame(screenshot_np, found, click_pt):
    """Сохраняет кадр с боксами и точкой клика (первые DEBUG_FRAME_LIMIT штук)."""
    global _debug_saved
    if not DEBUG or _debug_saved >= DEBUG_FRAME_LIMIT:
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    vis = screenshot_np.copy()
    for name, (conf, box) in found.items():
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{name} {conf:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if click_pt is not None:
        cv2.circle(vis, (int(click_pt[0]), int(click_pt[1])), 18, (0, 0, 255), 3)
    path = os.path.join(DEBUG_DIR, f"frame_{_debug_saved:03d}.png")
    cv2.imwrite(path, vis)
    _debug_saved += 1


def click_upgrades(found):
    """Кликает по центру каждого улучшения снизу вверх. Возвращает число кликов."""
    upgrades = [
        (name, data[1]) for name, data in found.items()
        if name in UPGRADE_CLASSES and data[0] > CONF_UPGRADE
    ]
    if not upgrades:
        return 0
    # снизу вверх: самый нижний (больший Y) кликается первым
    upgrades.sort(key=lambda item: box_center(item[1])[1], reverse=True)
    for name, box in upgrades:
        cx, cy = box_center(box)
        click_at(cx, cy)
        print(f"Покупка {name}: ({cx:.0f}, {cy:.0f})")
        time.sleep(0.25)
    return len(upgrades)


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

        # 2. Дополнительные действия (НЕ прерывают клики по печеньке)

        # 2a. Закрыть рекламу (только уверенно и не чаще кулдауна)
        close = found.get('close')
        if close and close[0] > CONF_CLOSE and now - last_close_time > CLOSE_COOLDOWN:
            cx, cy = box_center(close[1])
            click_at(cx, cy)
            last_close_time = now
            print(f"Закрытие рекламы: ({cx:.0f}, {cy:.0f})")
            time.sleep(0.15)

        # 2b. Собрать бонусную печеньку
        bonus = found.get('bonus')
        if bonus and bonus[0] > CONF_BONUS and now - last_bonus_time > BONUS_COOLDOWN:
            cx, cy = box_center(bonus[1])
            click_at(cx, cy)
            last_bonus_time = now
            print(f"Бонус собран: ({cx:.0f}, {cy:.0f})")
            time.sleep(0.1)

        # 2c. Раз в UPGRADE_INTERVAL секунд купить улучшения снизу вверх
        if now - last_upgrade_time >= UPGRADE_INTERVAL:
            if click_upgrades(found) > 0:
                last_upgrade_time = now

        # 3. ГЛАВНОЕ ДЕЙСТВИЕ — всегда кликаем по печеньке
        game = found.get('game')
        if game and game[0] > CONF_GAME:
            # свежая позиция — обновляем память
            last_game_box = game[1]
            last_game_time = now
            game_conf = game[0]
        else:
            # печеньку не увидели — используем запомненную позицию
            game_conf = 0.0

        if last_game_box is not None and now - last_game_time <= GAME_MEMORY:
            cx, cy = box_center(last_game_box)
            save_debug_frame(screenshot_np, found, (cx, cy))
            for _ in range(CLICKS_PER_BURST):
                pyautogui.click(cx, cy)
                time.sleep(CLICK_DELAY)
            if DEBUG:
                print(f"[cookie] {CLICKS_PER_BURST} кликов в ({cx:.0f}, {cy:.0f}) "
                      f"conf={game_conf:.2f}")
        else:
            # печеньку так и не нашли — короткая пауза
            time.sleep(0.05)


if __name__ == '__main__':
    print("Запуск. Для остановки уведите мышь в левый верхний угол.")
    if DEBUG:
        print(f"Режим отладки: печатаю детекции и сохраняю кадры в {DEBUG_DIR}/")
    main_loop()
