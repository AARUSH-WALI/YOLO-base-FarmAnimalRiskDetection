"""
Data Preparation & Annotation Pipeline for Monkey Custom Dataset
Prepares YOLO dataset structure, generates bounding box annotations (.txt),
creates train/val split and data.yaml configuration.
"""
import os
import glob
import shutil
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, 'datasets', 'monkey')

# Ground-truth normalized annotations [class_id, x_center, y_center, width, height]
# class_id: 0 = Monkey
ANNOTATIONS = {
    'i.jpg': [[0, 0.50, 0.52, 0.95, 0.90]],
    'images (1).jpg': [[0, 0.47, 0.52, 0.64, 0.83]],
    'images (2).jpg': [[0, 0.56, 0.54, 0.88, 0.91]],
    'images (3).jpg': [[0, 0.34, 0.53, 0.34, 0.76], [0, 0.70, 0.55, 0.35, 0.75]],
    'images (4).jpg': [[0, 0.68, 0.59, 0.44, 0.67], [0, 0.36, 0.61, 0.29, 0.64]],
    'images.jpg': [[0, 0.48, 0.52, 0.73, 0.95]],
    'mage-examples-from-the-Auxiliary-Datasets-On-the-top-row-two-examples-from-Human_Q320.webp': [
        [0, 0.53, 0.49, 0.43, 0.78]
    ],
    'pexels-jerico-stokovic-lee-3560352-scaled-e1599710263286.webp': [
        [0, 0.47, 0.50, 0.41, 0.74]
    ],
    'thumb (1).jpg': [[0, 0.50, 0.51, 0.32, 0.91]],
    'thumb.jpg': [[0, 0.45, 0.61, 0.87, 0.78]],
}

def setup_dataset():
    train_img_dir = os.path.join(DATASET_ROOT, 'images', 'train')
    val_img_dir = os.path.join(DATASET_ROOT, 'images', 'val')
    train_lbl_dir = os.path.join(DATASET_ROOT, 'labels', 'train')
    val_lbl_dir = os.path.join(DATASET_ROOT, 'labels', 'val')

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    src_dir = os.path.join(BASE_DIR, 'MonkeyDataset')
    files = list(ANNOTATIONS.keys())

    # Split 8 train / 2 val
    val_files = {'images (4).jpg', 'thumb (1).jpg'}

    for idx, fname in enumerate(files):
        src_path = os.path.join(src_dir, fname)
        if not os.path.exists(src_path):
            continue

        is_val = fname in val_files
        target_img_dir = val_img_dir if is_val else train_img_dir
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        # Standardize name (no spaces)
        clean_name = f"monkey_{idx:03d}.jpg"
        target_img_path = os.path.join(target_img_dir, clean_name)
        target_lbl_path = os.path.join(target_lbl_dir, clean_name.replace('.jpg', '.txt'))

        # Convert and save as JPG
        with Image.open(src_path) as im:
            im = im.convert('RGB')
            im.save(target_img_path, 'JPEG', quality=95)

        # Write YOLO label file
        boxes = ANNOTATIONS[fname]
        with open(target_lbl_path, 'w') as lf:
            for b in boxes:
                lf.write(f"{b[0]} {b[1]:.4f} {b[2]:.4f} {b[3]:.4f} {b[4]:.4f}\n")

        print(f"Processed: {fname} -> {clean_name} ({'val' if is_val else 'train'})")

    # Generate data.yaml
    yaml_content = f"""path: {DATASET_ROOT.replace('\\\\', '/').replace('\\', '/')}
train: images/train
val: images/val

nc: 1
names: ['Monkey']
"""
    yaml_path = os.path.join(DATASET_ROOT, 'data.yaml')
    with open(yaml_path, 'w') as yf:
        yf.write(yaml_content)

    print(f"\n[OK] Dataset generated at: {DATASET_ROOT}")
    print(f"[OK] data.yaml created at: {yaml_path}")
    return yaml_path

if __name__ == '__main__':
    setup_dataset()
