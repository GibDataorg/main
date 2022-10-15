import cv2 as cv2
from Classification import get_good_or_bad_type, get_type, Classification


def drawBBox(name, path, bboxes, c_type):
    bboxes = bboxes.cpu().detach().numpy().tolist()
    img = cv2.imread(path)
    print("Визуализируем", name)

    for bbox in bboxes:
        visualize_bbox(name, img, bbox, c_type)

def visualize_bbox(name, img, bbox, c_type):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    size = max(x_max - x_min, y_max - y_min)
    rec = 720 // ((y_max + y_min) // 2) * size

    if c_type:
        color, text = get_good_or_bad_type(rec)
    else:
        color, text = get_type(rec)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=2)

    ((text_width, text_height), _) = cv2.getTextSize("Stone", cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)

    cv2.putText(
        img,
        text=text,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=Classification.TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )

    cv2.imwrite('results/result_' + str(c_type) + "_" + name + '_.jpg', img)