from ultralytics import YOLO

# Загрузите обученную модель
model = YOLO("runs/segment/train/weights/best.pt")

# Запустите проверку
results = model.val(data="sonar-seg.yaml")

# Получите метрики для bounding boxes
box_metrics = results.box

# Расчет F1-меры для каждого класса
for class_id, class_name in enumerate(results.names):
    precision = box_metrics.p[class_id]  # Precision для класса
    recall = box_metrics.r[class_id]  # Recall для класса

    # Рассчитайте F1-меру
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print(f"Класс {class_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")
