import cv2
import numpy as np
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, enable_mkldnn=False, lang='en')
img = np.ones((100, 100, 3), dtype=np.uint8) * 255
res = ocr.ocr(img)
if res and isinstance(res[0], dict):
    print("Keys:", res[0].keys())
