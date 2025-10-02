#!/usr/bin/env python3
"""
Annotation Tool for QR Code Detection
Allows manual annotation of QR codes in images
Creates annotations in the required format
"""

import json
import cv2
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


class QRAnnotationTool:
    """Interactive QR code annotation tool"""
    
    def __init__(self, image_dir: str, output_file: str):
        self.image_dir = Path(image_dir)
        self.output_file = output_file
        self.annotations = []
        self.current_image = None
        self.current_image_path = None
        self.current_boxes = []  # list of [x1, y1, x2, y2]
        self.temp_box = None
        self.drawing = False
        self.start_point = None
        # BBox edit mode state
        self.edit_mode = False  # enable dragging corners of the last box
        self.dragging_vertex = False
        self.selected_vertex_idx: int = -1
        self.vertex_pick_radius = 10
        # Helper mode: free-corner visual alignment (saves bbox only)
        self.helper_mode = False
        self.helper_quad: List[Tuple[int, int]] = []
        
        # Load existing annotations if file exists
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                self.annotations = json.load(f)
            print(f"Loaded {len(self.annotations)} existing annotations")
        
        # Get image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            self.image_files.extend(list(self.image_dir.glob(f'*{ext}')))
            self.image_files.extend(list(self.image_dir.glob(f'*{ext.upper()}')))
        
        self.image_files = sorted(self.image_files)
        self.current_index = 0
        
        print(f"Found {len(self.image_files)} images")
        print("\nControls:")
        print("  Left click + drag: Draw bounding box")
        print("  'e': Toggle edit mode (drag corners of the last bbox)")
        print("  'h': Shape helper (drag 4 corners to match tilt; saves bbox only)")
        print("  'n': Next image")
        print("  'p': Previous image")
        print("  'u': Undo last box")
        print("  's': Save annotations")
        print("  'q': Quit (saves automatically)")
        print()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing and editing bounding boxes"""
        # Helper mode: drag any of the 4 helper vertices
        if self.helper_mode and self.helper_quad:
            if event == cv2.EVENT_LBUTTONDOWN:
                idx = self._find_nearest_point(self.helper_quad, (x, y))
                if idx is not None:
                    self.selected_vertex_idx = idx
                    self.dragging_vertex = True
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging_vertex:
                self.helper_quad[self.selected_vertex_idx] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.dragging_vertex:
                self.dragging_vertex = False
                self.selected_vertex_idx = -1
            return
        
        # Edit mode: drag nearest corner of the last bbox
        if self.edit_mode and self.current_boxes:
            last_bbox = self._normalize_bbox(self.current_boxes[-1])
            if event == cv2.EVENT_LBUTTONDOWN:
                idx = self._find_nearest_bbox_vertex(last_bbox, (x, y))
                if idx is not None:
                    self.selected_vertex_idx = idx
                    self.dragging_vertex = True
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging_vertex:
                self._update_bbox_vertex(-1, self.selected_vertex_idx, (x, y))
            elif event == cv2.EVENT_LBUTTONUP and self.dragging_vertex:
                self.dragging_vertex = False
                self.selected_vertex_idx = -1
            return
        
        # Normal box drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_box = (self.start_point[0], self.start_point[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                x_min = min(self.start_point[0], end_point[0])
                y_min = min(self.start_point[1], end_point[1])
                x_max = max(self.start_point[0], end_point[0])
                y_max = max(self.start_point[1], end_point[1])
                if (x_max - x_min) > 5 and (y_max - y_min) > 5:
                    self.current_boxes.append([x_min, y_min, x_max, y_max])
                self.temp_box = None
    
    def draw_boxes(self, image):
        """Draw all bounding boxes on image"""
        display_image = image.copy()
        
        # Draw confirmed boxes (axis-aligned)
        for bbox in self.current_boxes:
            x_min, y_min, x_max, y_max = map(int, self._normalize_bbox(bbox))
            cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), 
                         (0, 255, 0), 2)
            cv2.putText(display_image, "QR", (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # If editing, show corner handles
            if self.edit_mode and bbox is self.current_boxes[-1]:
                for (cx, cy) in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]:
                    cv2.circle(display_image, (int(cx), int(cy)), 4, (0, 255, 255), -1)
        
        # Draw temporary box while drawing
        if self.temp_box:
            x_min, y_min, x_max, y_max = self.temp_box
            cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max),
                         (255, 0, 0), 2)
        
        # Draw helper overlay quad (visual only)
        if self.helper_mode and self.helper_quad:
            pts = self.helper_quad
            for i in range(4):
                p1 = pts[i]
                p2 = pts[(i + 1) % 4]
                cv2.line(display_image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)
            for (qx, qy) in pts:
                cv2.circle(display_image, (int(qx), int(qy)), 4, (0, 255, 255), -1)
        
        # Add info text
        edit_text = " | EDIT: ON" if self.edit_mode else ""
        helper_text = " | HELPER: ON" if self.helper_mode else ""
        info_text = f"Image {self.current_index + 1}/{len(self.image_files)} | QR Codes: {len(self.current_boxes)}{edit_text}{helper_text}"
        cv2.putText(display_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_image
    
    def load_image(self):
        """Load current image and existing annotations"""
        if self.current_index < 0:
            self.current_index = 0
        if self.current_index >= len(self.image_files):
            self.current_index = len(self.image_files) - 1
        
        self.current_image_path = self.image_files[self.current_index]
        self.current_image = cv2.imread(str(self.current_image_path))
        
        # Load existing annotations for this image
        img_name = self.current_image_path.stem
        self.current_boxes = []
        self.edit_mode = False
        self.helper_mode = False
        self.helper_quad = []
        self.dragging_vertex = False
        self.selected_vertex_idx = -1
        
        for ann in self.annotations:
            if ann['image_id'] == img_name:
                boxes = []
                for qr in ann.get('qrs', []):
                    if isinstance(qr, dict) and 'bbox' in qr:
                        boxes.append(qr['bbox'])
                    else:
                        boxes.append(qr)
                self.current_boxes = boxes
                break
        
        print(f"Loaded: {self.current_image_path.name} ({len(self.current_boxes)} QR codes)")
    
    def save_current_annotations(self):
        """Save current image annotations"""
        img_name = self.current_image_path.stem
        
        # Remove existing annotation for this image
        self.annotations = [ann for ann in self.annotations 
                           if ann['image_id'] != img_name]
        
        # Add current annotations
        if self.current_boxes:
            annotation = {
                'image_id': img_name,
                'qrs': [{'bbox': [int(v) for v in box]} for box in self.current_boxes]
            }
            self.annotations.append(annotation)
    
    def save_to_file(self):
        """Save all annotations to file"""
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        print(f"\nAnnotations saved to: {self.output_file}")
        print(f"Total annotated images: {len(self.annotations)}")
        total_qrs = sum(len(ann['qrs']) for ann in self.annotations)
        print(f"Total QR codes: {total_qrs}")
    
    def run(self):
        """Run annotation tool"""
        if not self.image_files:
            print("No images found!")
            return
        
        cv2.namedWindow('QR Annotation Tool', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('QR Annotation Tool', self.mouse_callback)
        
        self.load_image()
        
        while True:
            display_image = self.draw_boxes(self.current_image)
            cv2.imshow('QR Annotation Tool', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                self.save_current_annotations()
                self.save_to_file()
                break
            
            elif key == ord('n'):
                # Next image
                self.save_current_annotations()
                self.current_index += 1
                if self.current_index >= len(self.image_files):
                    print("Last image reached!")
                    self.current_index = len(self.image_files) - 1
                else:
                    self.load_image()
            
            elif key == ord('p'):
                # Previous image
                self.save_current_annotations()
                self.current_index -= 1
                if self.current_index < 0:
                    print("First image reached!")
                    self.current_index = 0
                else:
                    self.load_image()
            
            elif key == ord('u'):
                # Undo last box
                if self.helper_mode and self.helper_quad:
                    # cancel helper overlay without changing bbox
                    self.helper_mode = False
                    self.helper_quad = []
                    print("Cancelled helper overlay")
                elif self.current_boxes:
                    self.current_boxes.pop()
                    print("Removed last bounding box")
            
            elif key == ord('s'):
                # Save
                self.save_current_annotations()
                self.save_to_file()
            
            elif key == ord('e'):
                # Toggle edit mode for last bbox
                if self.current_boxes:
                    self.edit_mode = not self.edit_mode
                    self.dragging_vertex = False
                    self.selected_vertex_idx = -1
                    state = 'ON' if self.edit_mode else 'OFF'
                    print(f"BBox edit mode: {state}")
            
            elif key == ord('h'):
                # Toggle helper shape mode (visual quad; saves bbox only)
                if not self.helper_mode:
                    if self.current_boxes:
                        x1, y1, x2, y2 = self._normalize_bbox(self.current_boxes[-1])
                        self.helper_quad = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        self.helper_mode = True
                        self.dragging_vertex = False
                        self.selected_vertex_idx = -1
                        print("Helper mode: ON (adjust corners, press 'h' again to apply)")
                else:
                    # Apply helper: convert quad to axis-aligned bbox and update last box
                    xs = [p[0] for p in self.helper_quad]
                    ys = [p[1] for p in self.helper_quad]
                    new_bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                    self.current_boxes[-1] = new_bbox
                    self.helper_mode = False
                    self.helper_quad = []
                    print("Helper applied: bbox updated")
        
        cv2.destroyAllWindows()

    def _normalize_bbox(self, item) -> List[int]:
        # Accept either list [x1,y1,x2,y2] or dict {'bbox': [...]}
        if isinstance(item, dict):
            return [int(v) for v in item.get('bbox', [0, 0, 0, 0])]
        return [int(v) for v in item]

    def _find_nearest_bbox_vertex(self, bbox: List[int], point: Tuple[int, int]):
        px, py = point
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        best_idx = None
        best_dist2 = self.vertex_pick_radius * self.vertex_pick_radius
        for i, (cx, cy) in enumerate(corners):
            dx = cx - px
            dy = cy - py
            d2 = dx*dx + dy*dy
            if d2 <= best_dist2:
                best_idx = i
                best_dist2 = d2
        return best_idx

    def _update_bbox_vertex(self, idx_in_boxes: int, vertex_idx: int, point: Tuple[int, int]):
        x, y = point
        bbox = self._normalize_bbox(self.current_boxes[idx_in_boxes])
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if 0 <= vertex_idx < 4:
            corners[vertex_idx] = (x, y)
            # Reconstruct bbox from updated corners (min/max)
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            new_bbox = [min(xs), min(ys), max(xs), max(ys)]
            self.current_boxes[idx_in_boxes] = new_bbox

    def _find_nearest_point(self, points: List[Tuple[int, int]], point: Tuple[int, int]):
        px, py = point
        best_idx = None
        best_dist2 = self.vertex_pick_radius * self.vertex_pick_radius
        for i, (cx, cy) in enumerate(points):
            dx = cx - px
            dy = cy - py
            d2 = dx*dx + dy*dy
            if d2 <= best_dist2:
                best_idx = i
                best_dist2 = d2
        return best_idx


def auto_detect_qrs(image_path: str) -> List[List[float]]:
    """
    Automatically detect QR codes using OpenCV
    Returns list of bounding boxes
    """
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try QR code detector
    detector = cv2.QRCodeDetector()
    retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(gray)
    
    boxes = []
    if retval and points is not None:
        for point_set in points:
            # Get bounding box from points
            x_coords = [p[0] for p in point_set]
            y_coords = [p[1] for p in point_set]
            
            x_min = int(min(x_coords))
            y_min = int(min(y_coords))
            x_max = int(max(x_coords))
            y_max = int(max(y_coords))
            
            boxes.append([x_min, y_min, x_max, y_max])
    
    return boxes


def auto_annotate_directory(image_dir: str, output_file: str):
    """
    Automatically detect and annotate QR codes in all images
    """
    image_path = Path(image_dir)
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(image_path.glob(f'*{ext}')))
        image_files.extend(list(image_path.glob(f'*{ext.upper()}')))
    
    annotations = []
    
    print(f"Auto-detecting QR codes in {len(image_files)} images...")
    
    for img_file in image_files:
        boxes = auto_detect_qrs(str(img_file))
        
        if boxes:
            annotation = {
                'image_id': img_file.stem,
                'qrs': [{'bbox': box} for box in boxes]
            }
            annotations.append(annotation)
            print(f"  {img_file.name}: {len(boxes)} QR codes detected")
    
    # Save annotations
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\nAuto-annotations saved to: {output_file}")
    print(f"Total images with QR codes: {len(annotations)}")
    total_qrs = sum(len(ann['qrs']) for ann in annotations)
    print(f"Total QR codes detected: {total_qrs}")


def main():
    parser = argparse.ArgumentParser(
        description='QR Code Annotation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Manual annotation
  python annotate.py --images QR_Dataset/train_images --output annotations.json
  
  # Auto-detection (creates initial annotations)
  python annotate.py --images QR_Dataset/train_images --output annotations.json --auto
        """
    )
    
    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing images to annotate')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for annotations')
    parser.add_argument('--auto', action='store_true',
                       help='Automatic QR detection (no manual annotation)')
    
    args = parser.parse_args()
    
    if args.auto:
        auto_annotate_directory(args.images, args.output)
    else:
        tool = QRAnnotationTool(args.images, args.output)
        tool.run()


if __name__ == '__main__':
    main()

