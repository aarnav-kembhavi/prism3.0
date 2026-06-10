import cv2
import numpy as np
import sys
import os

def rotate_and_pad(image_path, angles=[5, 15, 25, 45]):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return

    # Setup output paths
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.dirname(image_path)
    
    h, w = img.shape[:2]
    
    for angle in angles:
        # Get the rotation matrix
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate the bounding box of the rotated image so we don't cut off corners
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust the rotation matrix translation to keep the image centered
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Add 150px of extra padding to simulate a "dark desk" around the document
        pad = 150
        final_w = new_w + (pad * 2)
        final_h = new_h + (pad * 2)
        M[0, 2] += pad
        M[1, 2] += pad
        
        # Perform the rotation and padding
        # borderValue=(40, 40, 40) fills the background with a realistic dark gray
        rotated = cv2.warpAffine(img, M, (final_w, final_h), borderValue=(40, 40, 40))
        
        out_path = os.path.join(out_dir, f"{base_name}_rotated_{angle}deg.png")
        cv2.imwrite(out_path, rotated)
        print(f"Generated: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python augment_rotations.py <path_to_perfect_image>")
    else:
        rotate_and_pad(sys.argv[1])
