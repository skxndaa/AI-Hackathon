#!/usr/bin/env python3
"""
QR Code Detection Inference Script
Runs inference on a folder of images and outputs results in JSON format
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


def run_inference(model_path, input_dir, output_json, conf_threshold=0.25, save_images=False, save_dir=None):
    """
    Runs QR code detection inference on a directory of images.
    
    Args:
        model_path: Path to trained YOLO model weights (.pt file)
        input_dir: Directory containing images to process
        output_json: Output path for JSON results file
        conf_threshold: Confidence threshold for detections (0-1)
        save_images: Whether to save images with bounding boxes
        save_dir: Directory to save annotated images (if save_images=True)
    
    Returns:
        List of detection results
    """
    print(f"\nLoading trained model from: {model_path}")
    
    # Load the trained YOLO model
    model = YOLO(model_path)
    
    # Get all image files
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
        image_files.extend(list(input_path.glob(f'*{ext.upper()}')))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f" No images found in {input_dir}")
        return []
    
    print(f"Found {len(image_files)} images to process")
    
    # Prepare submission data
    submission_data = []
    
    # Process each image
    for img_path in tqdm(image_files, desc=" Detecting QR codes"):
        # Run prediction
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            verbose=False
        )
        
        result = results[0]
        image_id = img_path.stem
        qrs_in_image = []
        
        # Extract bounding boxes
        if result.boxes:
            for box in result.boxes:
                # Get bounding box coordinates [x_min, y_min, x_max, y_max]
                bbox_coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                
                # Get confidence score
                confidence = float(box.conf[0].cpu().numpy())
                
                qr_dict = {
                    "bbox": bbox_coords,
                    "confidence": confidence
                }
                
                qrs_in_image.append(qr_dict)
        
        # Add to submission
        image_dict = {
            "image_id": image_id,
            "qrs": qrs_in_image
        }
        
        submission_data.append(image_dict)
    
    # Save results to JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(submission_data, f, indent=2)
    
    print(f"\nInference complete!")
    print(f"Results saved to: {output_json}")
    
    # Print summary statistics
    total_qrs = sum(len(img['qrs']) for img in submission_data)
    images_with_qrs = sum(1 for img in submission_data if len(img['qrs']) > 0)
    
    print(f"\nSummary:")
    print(f"   Total images processed: {len(submission_data)}")
    print(f"   Images with QR codes: {images_with_qrs}")
    print(f"   Total QR codes detected: {total_qrs}")
    if images_with_qrs > 0:
        print(f"   Average QR codes per image: {total_qrs / images_with_qrs:.2f}")
    
    # Optionally save annotated images
    if save_images:
        if save_dir is None:
            save_dir = output_path.parent / 'annotated_images'
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving annotated images to: {save_dir}")
        
        for img_path in tqdm(image_files, desc="Saving annotated images"):
            results = model.predict(
                source=str(img_path),
                conf=conf_threshold,
                save=True,
                project=str(save_dir.parent),
                name=save_dir.name,
                exist_ok=True,
                verbose=False
            )
    
    return submission_data


def main():
    parser = argparse.ArgumentParser(
        description='Run QR code detection inference on images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python infer.py --input demo_images/ --output submission.json
  
  # Specify custom model
  python infer.py --input test_images/ --output results.json --model runs/qr_detection_model/weights/best.pt
  
  # Lower confidence threshold and save annotated images
  python infer.py --input test_images/ --output results.json --conf 0.2 --save-images
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file path for results')
    parser.add_argument('--model', type=str, default='runs/qr_detection_model/weights/best.pt',
                       help='Path to trained model weights (default: runs/qr_detection_model/weights/best.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--save-images', action='store_true',
                       help='Save annotated images with bounding boxes')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save annotated images (default: output_dir/annotated_images)')
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model).exists():
        print(f" Error: Model weights not found at '{args.model}'")
        print("Please train a model first using train.py or specify a valid model path")
        return
    
    # Run inference
    try:
        run_inference(
            model_path=args.model,
            input_dir=args.input,
            output_json=args.output,
            conf_threshold=args.conf,
            save_images=args.save_images,
            save_dir=args.save_dir
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        raise


if __name__ == '__main__':
    main()

