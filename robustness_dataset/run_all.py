import os
import sys
from augment_rotations import rotate_and_pad

base_dir = r"d:\DEVELOPMENT\prism3.0\robustness_dataset"
print(f"Starting rotation augmentation in {base_dir}")

count = 0
for root, dirs, files in os.walk(base_dir):
    for f in files:
        if f == 'glared.png':
            img_path = os.path.join(root, f)
            print(f"Processing {img_path}...")
            rotate_and_pad(img_path)
            count += 1

print(f"Finished! Processed {count} glared.png files.")
