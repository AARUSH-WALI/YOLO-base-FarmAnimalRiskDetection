"""
Train Custom YOLOv8 Model on Monkey Dataset
Uses transfer learning from yolov8n.pt to specialize on monkey detection.
"""
import os
from ultralytics import YOLO

def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, 'datasets', 'monkey', 'data.yaml')

    print("=" * 60)
    print("  🐒 Starting YOLOv8 Fine-Tuning for Monkey Detection")
    print(f"  📁 Dataset config: {yaml_path}")
    print("=" * 60)

    # Load base nano model
    model = YOLO('yolov8n.pt')

    # Train for 15 epochs (fast transfer learning suitable for CPU demo)
    results = model.train(
        data=yaml_path,
        epochs=15,
        imgsz=416,
        batch=4,
        workers=0,
        project='runs/train',
        name='monkey_detector',
        exist_ok=True,
        verbose=True
    )

    best_weights = os.path.join('runs', 'train', 'monkey_detector', 'weights', 'best.pt')
    print("=" * 60)
    print("  ✅ Fine-tuning completed!")
    print(f"  🏆 Best model saved at: {best_weights}")
    print("=" * 60)

if __name__ == '__main__':
    train()
