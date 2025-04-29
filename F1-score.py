from ultralytics import YOLO

model = YOLO("runs/segment/train/weights/best.pt")

results = model.val(data="sonar-seg.yaml")

box_metrics = results.box

for class_id, class_name in enumerate(results.names):
    precision = box_metrics.p[class_id]
    recall = box_metrics.r[class_id]

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print(f"Класс {class_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")
