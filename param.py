from ultralytics import YOLO

# Модель
model = YOLO('runs/segment/train/weights/best.pt')

# Тестовые изображения
source = 'test'

# Сохрвнение результатов
project = '/Users/imac/Desktop/Diplom/projects/new_data2'
name = 'predict'


results = model.predict(
    source=source,
    save=True,
    project=project,
    name=name,
    exist_ok=True
)
