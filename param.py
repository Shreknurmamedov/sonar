from ultralytics import YOLO

# Загрузка модели
model = YOLO('runs/segment/train/weights/best.pt')

# Указание пути к тестовым изображениям
source = 'test'

# Указание пути для сохранения результатов
project = '/Users/imac/Desktop/Diplom/projects/new_data2'  # Проектная директория
name = 'predict'  # Имя папки внутри проектной директории

# Запуск предсказаний с указанием пути сохранения
results = model.predict(
    source=source,
    save=True,
    project=project,
    name=name,
    exist_ok=True  # Разрешить перезапись существующей папки
)
