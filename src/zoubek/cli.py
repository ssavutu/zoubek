
from zoubek import ocr
from zoubek.preprocess import preprocess


def main():
    pages = preprocess("./gregg.pdf")
    engine = ocr.init_paddle()
    for number, page in enumerate(pages):
        ocr.isolate_shorthand_and_text(page, number, engine)


if __name__ == "__main__":
    main()
