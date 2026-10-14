# AI Cookie Clicker — как запустить

## 1. Установка

Откройте терминал в папке проекта и выполните:

```
py -3.12 -m venv venv
venv\Scripts\activate
pip install ultralytics pyautogui mss albumentations opencv-python pillow
```

## 2. Обучение модели

```
python train.py
```

Готовая модель появится в `runs/detect/cookie_detector/weights/best.pt`.

## 3. Запуск бота (автокликера)

```
python app.py
```

Бот быстро и непрерывно кликает по печеньке. Он ненадолго прерывается только чтобы:
- закрыть рекламу,
- собрать бонусную печеньку,
- купить улучшение.

**Остановка:** уведите мышь в левый верхний угол экрана.

## 4. Автосбор скриншотов (Задача 5)

```
python screener.py
```

Делает скриншот каждые 5 секунд в папку `screenshots/`.
**Остановка:** `Ctrl+C`.

## 5. Увеличение датасета

```
python augment.py
```

Создаёт аугментированные копии из `screenshots/` в `screenshots_augmented/`.

## 6. Авторазметка в X-AnyLabeling

```
python ptToOnnx.py
```

Затем в X-AnyLabeling:
1. Откройте папку с картинками (File → Open Dir).
2. На левой панели нажмите кнопку **AI** (Auto Labeling).
3. Нажмите **+** (Load Custom Model) и выберите файл `config.yaml`.
4. Нажимайте **D** и **I** — модель сама расставит боксы.

Классы модели: `game`, `close`, `bonus`, `upgrade`.

## 7. Экспорт разметки в YOLO

В X-AnyLabeling: меню **Export** → **YOLO HBB** → выбрать папку проекта →
галочки **Save with images** и **Skip empty labels** → **OK**.
