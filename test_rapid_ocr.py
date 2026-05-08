import cv2
import numpy as np
from rapid_latex_ocr import LatexOCR

def test():
    try:
        # Create a dummy image with noise instead of solid white
        dummy_img = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        cv2.imwrite("dummy_test.jpg", dummy_img)
        
        with open("dummy_test.jpg", "rb") as f:
            img_bytes = f.read()

        model = LatexOCR()
        res, elapse = model(img_bytes)
        print("Result:", res)
        print("Elapse:", elapse)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
