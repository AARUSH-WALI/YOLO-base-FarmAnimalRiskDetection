"""
Train Custom YOLOv8 Model on Roboflow Monkey & Person Dataset (5,355 images)
Includes negative samples (Person, Background) to eliminate false positives on humans and fans.
"""
import os
from ultralytics import YOLO

def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, 'datasets', 'monkey_roboflow', 'data.yaml')

    print("=" * 65)
    print("  🐒 Starting YOLOv8 Training on Roboflow Dataset (5,355 images)")
    print(f"  📁 Dataset config: {yaml_path}")
    print("=" * 65)

    model = YOLO('yolov8n.pt')

    # Train for 5 epochs on CPU with batch 8 and image size 416
    results = model.train(
        data=yaml_path,
        epochs=5,
        imgsz=416,
        batch=8,
        workers=0,
        project='runs/train',
        name='monkey_roboflow_detector',
        exist_ok=True,
        verbose=True
    )

    best_weights = os.path.join('runs', 'train', 'monkey_roboflow_detector', 'weights', 'best.pt')
    print("=" * 65)
    print("  ✅ Training completed!")
    print(f"  🏆 Best weights saved at: {best_weights}")
    print("=" * 65)

if __name__ == '__main__':
    train()
