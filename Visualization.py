import cv2 as cv2
from Classification import *


def draw_and_count_box(name, path, bboxes, c_type):
    bboxes = bboxes.cpu().detach().numpy().tolist()
    img = cv2.imread(path)
    print("Визуализируем", name)

    for bbox in bboxes:
        operate_bbox(name, img, bbox, c_type)


def operate_bbox(name, img, bbox, c_type):  # c_type = 0 - types, 1 - good/bad
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    size = max(x_max-x_min, y_max-y_min)
    rec = (720 / ((y_min + y_max) / 2))

    print(size, rec * size)
    size = int(rec * size)

    update_max_size(size)

    if c_type:
        color, text, text_color = make_type_class(size)
    else:
        color, text, text_color = make_type_good_or_bad(size)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=2)

    ((text_width, text_height), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.5 * text_height)), (x_min + text_width, y_min), color, -1)

    cv2.putText(
        img,
        text=text,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=text_color,
        lineType=cv2.LINE_AA,
    )

    cv2.imwrite('results/result_' + str(c_type) + "_" + name + '_.jpg', img)
