import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
from pycocotools.mask import encode

def json_to_coco(json_dir, output_dir, image_dir, split='train'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    coco_data = {
        "info": {
            "description": "Sonar Segmentation Dataset",
            "version": "1.0",
            "year": 2023,
            "contributor": "Your Name",
            "url": "http://example.com",
            "date_created": "2023-10-01"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-NoDerivs License",
                "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/"
            }
        ],
        "categories": [
            {
                "id": 0,
                "name": "obstacle",
                "supercategory": "object"
            },
            {
                "id": 1,
                "name": "ground",
                "supercategory": "object"
            }
        ],
        "images": [],
        "annotations": []
    }

    image_id = 0
    annotation_id = 0

    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)

            image_path = os.path.join(image_dir, data['imagePath'])
            image = Image.open(image_path)
            width, height = image.size

            coco_data["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": data['imagePath']
            })

            for shape in data['shapes']:
                label = shape['label']
                points = shape['points']

                if label == 'obstacle':
                    category_id = 0
                elif label == 'ground':
                    category_id = 1
                else:
                    continue

                segmentation = []
                for i in range(0, len(points)):
                    segmentation.extend([points[i][0], points[i][1]])

                segmentation_np = np.array(segmentation).reshape(-1, 2).astype(np.float32)
                rles = encode(np.asfortranarray(segmentation_np[:, 1::-1] < [width, height]))
                area = float(rles['size'][0] * rles['size'][1])
                bbox = [
                    float(min(segmentation_np[:, 0])),
                    float(min(segmentation_np[:, 1])),
                    float(max(segmentation_np[:, 0]) - min(segmentation_np[:, 0])),
                    float(max(segmentation_np[:, 1]) - min(segmentation_np[:, 1]))
                ]

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })

                annotation_id += 1

            image_id += 1

    with open(os.path.join(output_dir, f'{split}.json'), 'w') as f:
        json.dump(coco_data, f, indent=4)

# Пути к директориям
train_json_dir = '/Users/imac/Desktop/Diplom/projects/new_data 2/json/train'
val_json_dir = '/Users/imac/Desktop/Diplom/projects/new_data 2/json/val'
output_dir = '/Users/imac/Desktop/Diplom/projects/new_data 2/annotations'
train_image_dir = '/Users/imac/Desktop/Diplom/projects/new_data 2/images/train'
val_image_dir = '/Users/imac/Desktop/Diplom/projects/new_data 2/images/val'

json_to_coco(train_json_dir, output_dir, train_image_dir, split='train')
json_to_coco(val_json_dir, output_dir, val_image_dir, split='val')
