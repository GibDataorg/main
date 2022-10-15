class Classification:

    BOX_COLOR1 = (139, 0, 0)  # DarkRed
    BOX_COLOR2 = (255, 0, 255)  # Fuchsia
    BOX_COLOR3 = (199, 21, 133)  # MediumVioletRed
    BOX_COLOR4 = (0, 0, 255)  # Blue
    BOX_COLOR5 = (30, 144, 255)  # DodgerBlue
    BOX_COLOR6 = (0, 191, 255)  # DeepSkyBlue
    BOX_COLOR7 = (135, 206, 250)  # LightSkyBlue

    BOX_COLOR_GOOD = (0, 255, 0)
    BOX_COLOR_BAD = (0, 0, 255) # Red (Негабарит)

    TEXT_COLOR = (255, 255, 255)  # White

    # Classification
    cl_1 = 100  # Негабарит
    types_7 = {"#1": [], "#2": [], "#3": [], "#4": [], "#5": [], "#6": [], "#7": []}
    overall_classification = {"good": [], "bad": []}



def get_good_or_bad_type(size):
    # Классификация по негабариту
    if size < Classification.cl_1:
        color = Classification.BOX_COLOR_GOOD
        text = "GOOD"
    else:
        color = Classification.BOX_COLOR_BAD
        text = "BAD"

    return color, text

def get_type(size):
    if size > 250:
        color = Classification.BOX_COLOR1
        text = "#1"
    elif size > 150 & size <= 250:
        color = Classification.BOX_COLOR2
        text = "#2"
    elif size > 100 & size <= 150:
        color = Classification.BOX_COLOR3
        text = "#3"
    elif size > 80 & size <= 100:
        color = Classification.BOX_COLOR4
        text = "#4"
    elif size > 70 & size <= 80:
        color = Classification.BOX_COLOR5
        text = "#5"
    elif size > 40 & size <= 70:
        color = Classification.BOX_COLOR6
        text = "#6"
    elif size > 0 & size <= 40:
        color = Classification.BOX_COLOR7
        text = "#7"

    return color, text


