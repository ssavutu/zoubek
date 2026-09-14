import os

_threads = str(max(1, (os.cpu_count() or 2) // 2))
os.environ.setdefault("OMP_NUM_THREADS", _threads)
os.environ.setdefault("MKL_NUM_THREADS", _threads)

import cv2
import numpy as np
from paddleocr import PaddleOCRVL


def init_paddle():
    engine = PaddleOCRVL(
        pipeline_version="v1.6",
        use_doc_orientation_classify=True,
        enable_mkldnn=False,
    )
    return engine

def isolate_shorthand_and_text(page: np.ndarray, number: int, engine):
    dir = "isolated_assets_page_" + str(number)
    os.makedirs(dir, exist_ok=True)

    if page.ndim == 2:
        page = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    result = engine.predict(page)[0]

    for index, block in enumerate(result["parsing_res_list"]):
        x1, y1, x2, y2 = block.bbox
        element_type = block.label

        match element_type:
            case "image" | "header_image" | "footer_image" | "table" | "chart" | "seal":
                cropped = page[y1:y2, x1:x2]
                cv2.imwrite(f"{dir}/{element_type}_{index}.jpg", cropped)
            case _:
                full_text = block.content
                with open(f"{dir}/{element_type}_{index}.txt", "w", encoding="utf-8") as file:
                    file.write(full_text)
