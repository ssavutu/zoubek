from collections.abc import Iterator

import cv2
import numpy as np
from pdf2image import convert_from_path as convert
from pdf2image import pdfinfo_from_path


def preprocess(path: str) -> Iterator[np.ndarray]:
    page_count = pdfinfo_from_path(path)["Pages"]
    for number in range(1, page_count + 1):
        page = convert(path, dpi=400, first_page=number, last_page=number)[0]
        img = np.array(page.convert("L"))
        yield sharpen(img)

def sharpen(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    upscaled = cv2.resize(img, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    binary_inverted = cv2.bitwise_not(upscaled)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) 
    closed = cv2.morphologyEx(binary_inverted, cv2.MORPH_CLOSE, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))
    closed2 = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel2)

    blurred = cv2.GaussianBlur(closed2, (5,5), 0)

    _, high_res = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)
    final = cv2.bitwise_not(high_res)

    return cv2.resize(final, (w, h), interpolation=cv2.INTER_AREA)

