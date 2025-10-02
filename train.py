#!/usr/bin/env python3
"""
QR Code Detection Model Training Script
Trains a YOLOv8 model on QR code detection dataset
"""

import json
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


def convert_bbox_to_yolo(img_width, img_height, bbox):
    """
    Converts [x_min, y_min, x_max, y_max] to YOLO's normalized format.
    
    Args:
        img_width: Image width in pixels
        img_height: Image height in pixels
        bbox: Bounding box in [x_min, y_min, x_max, y_max] format
    
    Returns:
        YOLO format string: "class_id x_center y_center width height" (normalized)
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    return f"0 {x_center_norm} {y_center_norm} {w_norm} {h_norm}\n"


def prepare_dataset(raw_image_dir, annotation_json_path, processed_data_dir, test_size=0.2):
    """
    Reads the JSON annotations, splits data, and creates YOLO-formatted files.
    
    Args:
        raw_image_dir: Directory containing raw training images
        annotation_json_path: Path to annotations.json file
        processed_data_dir: Output directory for processed data
        test_size: Fraction of data to use for validation (default: 0.2)
    """
    print("\n🔧 Starting dataset preparation...")
    
    raw_image_dir = Path(raw_image_dir)
    annotation_json_path = Path(annotation_json_path)
    processed_data_dir = Path(processed_data_dir)

    # Load annotations
    with open(annotation_json_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"📋 Loaded {len(annotations)} annotations")

    # Create output directories
    for subset in ['train', 'val']:
        (processed_data_dir / subset / 'images').mkdir(parents=True, exist_ok=True)
        (processed_data_dir / subset / 'labels').mkdir(parents=True, exist_ok=True)

    # Extract image IDs
    image_ids = []
    for anno in annotations:
        if 'image_id' in anno:
            image_ids.append(anno['image_id'])
        else:
            print(f"⚠️  Warning: Annotation missing 'image_id': {anno}. Skipping.")

    # Split data
    train_ids, val_ids = train_test_split(image_ids, test_size=test_size, random_state=42)
    print(f"📊 Splitting data: {len(train_ids)} training images, {len(val_ids)} validation images.")

    # Process each annotation
    for annotation in tqdm(annotations, desc="Processing annotations"):
        if 'image_id' not in annotation:
            continue

        image_id = annotation['image_id']
        subset = 'train' if image_id in train_ids else 'val'

        image_filename = f"{image_id}.jpg"
        source_image_path = raw_image_dir / image_filename

        if not source_image_path.exists():
            print(f"⚠️  Image not found for ID '{image_id}', skipping.")
            continue

        # Copy image to processed directory
        dest_image_path = processed_data_dir / subset / 'images' / image_filename
        shutil.copy(source_image_path, dest_image_path)

        # Get image dimensions
        with Image.open(source_image_path) as img:
            img_width, img_height = img.size

        # Create YOLO format label file
        label_path = processed_data_dir / subset / 'labels' / f"{image_id}.txt"
        with open(label_path, 'w') as label_file:
            if 'qrs' in annotation:
                for qr in annotation['qrs']:
                    yolo_string = convert_bbox_to_yolo(img_width, img_height, qr['bbox'])
                    label_file.write(yolo_string)
            else:
                print(f"⚠️  Warning: Annotation for image_id '{image_id}' has no 'qrs' field.")

    print("✅ Dataset preparation complete!")


def create_dataset_yaml(processed_data_dir, yaml_path):
    """
    Creates a YAML configuration file for YOLO training.
    
    Args:
        processed_data_dir: Directory containing processed training/validation data
        yaml_path: Output path for the YAML file
    """
    processed_data_dir = Path(processed_data_dir).resolve()
    yaml_path = Path(yaml_path)
    
    yaml_content = f"""# QR Code Detection Dataset Configuration
train: {processed_data_dir}/train/images
val: {processed_data_dir}/val/images

# Number of classes
nc: 1

# Class names
names: ['qr_code']
"""

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"📝 Created dataset YAML file at: {yaml_path}")


def train_model(yaml_path, epochs=50, batch_size=16, img_size=640, model_name='yolov8s.pt', 
                output_dir='runs', project_name='qr_detection_model'):
    """
    Trains a YOLO model for QR code detection.
    
    Args:
        yaml_path: Path to dataset YAML configuration
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        model_name: Pre-trained YOLO model to use
        output_dir: Directory to save training outputs
        project_name: Name of the training project
    """
    print(f"\n🚀 Starting model training with {model_name}...")
    
    # Load YOLO model
    model = YOLO(model_name)

    # Train the model
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=str(output_dir),
        name=project_name,
        patience=10,  # Early stopping patience
        save=True,
        val=True
    )

    print("✅ Training complete!")
    print(f"📁 Model weights saved to: {output_dir}/{project_name}/weights/")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 model for QR code detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--images', type=str, default='dataset/train_images',
                       help='Directory containing training images')
    parser.add_argument('--annotations', type=str, default='dataset/annotations.json',
                       help='Path to annotations JSON file')
    parser.add_argument('--output', type=str, default='processed_data',
                       help='Output directory for processed data')
    parser.add_argument('--yaml', type=str, default='qr_dataset.yaml',
                       help='Output path for dataset YAML file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                       help='Pre-trained YOLO model (default: yolov8s.pt)')
    parser.add_argument('--project-dir', type=str, default='runs',
                       help='Project directory for training outputs')
    parser.add_argument('--name', type=str, default='qr_detection_model',
                       help='Training run name')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip dataset preprocessing (use if already processed)')
    
    args = parser.parse_args()
    
    # Step 1: Prepare dataset (unless skipped)
    if not args.skip_preprocessing:
        prepare_dataset(
            raw_image_dir=args.images,
            annotation_json_path=args.annotations,
            processed_data_dir=args.output
        )
    else:
        print("⏭️  Skipping dataset preprocessing")
    
    # Step 2: Create YAML configuration
    create_dataset_yaml(
        processed_data_dir=args.output,
        yaml_path=args.yaml
    )
    
    # Step 3: Train model
    train_model(
        yaml_path=args.yaml,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        model_name=args.model,
        output_dir=args.project_dir,
        project_name=args.name
    )
    
    print("\n🎉 Training pipeline completed successfully!")
    print(f"💡 Best model weights: {args.project_dir}/{args.name}/weights/best.pt")


if __name__ == '__main__':
    main()

