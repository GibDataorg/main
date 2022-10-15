class Classification:
    BOX_COLOR1 = (0, 0, 139)  # DarkRed
    BOX_COLOR2 = (255, 0, 255)  # Fuchsia
    BOX_COLOR3 = (133, 21, 199)  # MediumVioletRed
    BOX_COLOR4 = (255, 0, 0)  # Blue
    BOX_COLOR5 = (255, 144, 30)  # DodgerBlue
    BOX_COLOR6 = (255, 191, 0)  # DeepSkyBlue
    BOX_COLOR7 = (250, 206, 135)  # LightSkyBlue

    BOX_COLOR_GOOD = (0, 255, 0)  # Green (GOOD)
    BOX_COLOR_BAD = (0, 0, 255)  # Red (BAD)

    TEXT_COLOR_BLACK = (0, 0, 0)
    TEXT_COLOR_WHITE = (255, 255, 255)

    # Classification
    type_classes = {
        "#1": {
            "count": 0,
            "sum": 0
        },
        "#2": {
            "count": 0,
            "sum": 0
        },
        "#3": {
            "count": 0,
            "sum": 0
        },
        "#4": {
            "count": 0,
            "sum": 0
        },
        "#5": {
            "count": 0,
            "sum": 0
        },
        "#6": {
            "count": 0,
            "sum": 0
        },
        "#7": {
            "count": 0,
            "sum": 0
        }
    }

    size_limit = 100  # Негабарит
    size_classes = {

        "GOOD": {
            "count": 0,
            "sum": 0
        },
        "BAD": {
            "count": 0,
            "sum": 0
        }
    }

    max_size = 0


def make_type_good_or_bad(size):
    # Классификация по негабариту
    if size < Classification.size_limit:
        color = Classification.BOX_COLOR_GOOD
        text = "GOOD"
        text_color = Classification.TEXT_COLOR_BLACK

    else:
        color = Classification.BOX_COLOR_BAD
        text = "BAD"
        text_color = Classification.TEXT_COLOR_WHITE

    Classification.size_classes[text]["count"] += 1
    Classification.size_classes[text]["sum"] += size

    return color, text, text_color


def make_type_class(size):
    if size <= 40:
        color = Classification.BOX_COLOR7
        text = "#7"
        text_color = Classification.TEXT_COLOR_BLACK

    elif size <= 70:
        color = Classification.BOX_COLOR6
        text = "#6"
        text_color = Classification.TEXT_COLOR_BLACK

    elif size <= 80:
        color = Classification.BOX_COLOR5
        text = "#5"
        text_color = Classification.TEXT_COLOR_BLACK

    elif size <= 100:
        color = Classification.BOX_COLOR4
        text = "#4"
        text_color = Classification.TEXT_COLOR_WHITE

    elif size <= 150:
        color = Classification.BOX_COLOR3
        text = "#3"
        text_color = Classification.TEXT_COLOR_BLACK

    elif size <= 250:
        color = Classification.BOX_COLOR2
        text = "#2"
        text_color = Classification.TEXT_COLOR_BLACK

    else:
        color = Classification.BOX_COLOR1
        text = "#1"
        text_color = Classification.TEXT_COLOR_WHITE

    Classification.type_classes[text]["count"] += 1
    Classification.type_classes[text]["sum"] += size

    return color, text + " | " + str(size), text_color


def update_max_size(size):
    if size > Classification.max_size:
        Classification.max_size = size
