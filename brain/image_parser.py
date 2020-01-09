import operator

import cv2
import numpy as np


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def parse_grid(img):
    cropped = crop_to_grid(img)
    enhanced_digits = enhance_digits(cropped)
    squares = infer_grid(enhanced_digits)
    patches = extract_patches(enhanced_digits, squares)
    digits = [extract_digit(patch) for patch in patches]
    return digits


def crop_to_grid(img):
    enhanced_grid = enhance_grid_lines(img)
    corners = find_corners_of_largest_polygon(enhanced_grid)
    cropped = crop_and_warp(img, corners)

    # Apply twice for good measure
    enhanced_grid = enhance_grid_lines(cropped)
    corners = find_corners_of_largest_polygon(enhanced_grid)
    cropped = crop_and_warp(cropped, corners)
    return cropped


def enhance_grid_lines(img):
    """Blur, threshold and dilate an image."""

    blur_fraction = 50.0
    neighbourhood_fraction = 45.0

    # Blur
    blur_size = make_odd(max(img.shape) / blur_fraction)
    proc = cv2.GaussianBlur(img.copy(), (blur_size, blur_size), 0)

    # Adaptive threshold
    neighbourhood_size = make_odd(max(img.shape) / neighbourhood_fraction)

    proc = cv2.adaptiveThreshold(
        proc,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        neighbourhood_size,
        2,
    )

    # Invert
    proc = cv2.bitwise_not(proc, proc)

    # Dilate
    proc = cv2.dilate(
        proc,
        np.array([[0.0, 0.5, 0.0], [0.5, 0.5, 0.5], [0.0, 0.5, 0.0]], dtype=np.uint8),
    )
    return proc


def enhance_digits(img):
    blur_fraction = 50.0
    neighbourhood_factor = 45.0

    # Blur
    blur_size = make_odd(max(img.shape) / blur_fraction)
    proc = cv2.GaussianBlur(img.copy(), (blur_size, blur_size), 0)

    # Adaptive threshold
    neighbourhood_size = make_odd(max(img.shape) / neighbourhood_factor)

    proc = cv2.adaptiveThreshold(
        proc,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        neighbourhood_size,
        5,
    )

    # Invert
    proc = cv2.bitwise_not(proc, proc)
    return proc


def make_odd(i):
    i = int(i)
    if i % 2 == 0:
        return i + 1
    return i


def find_corners_of_largest_polygon(img):
    """Finds the 4 extreme corners of the largest contour in the image."""
    contours, _ = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # Find contours
    contours = sorted(
        contours, key=cv2.contourArea, reverse=True
    )  # Sort by area, descending
    polygon = contours[0]  # Largest image

    # Use of `operator.itemgetter` with `max` and `min` allows us to get the index
    # of the point
    # Each point is an array of 1 coordinate, hence the [0] getter,
    # then [0] or [1] used to get x and y respectively.

    # Bottom-right point has the largest (x + y) value
    # Top-left has point smallest (x + y) value
    # Bottom-left point has smallest (x - y) value
    # Top-right point has largest (x - y) value
    bottom_right, _ = max(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1)
    )
    top_left, _ = min(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1)
    )
    bottom_left, _ = min(
        enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1)
    )
    top_right, _ = max(
        enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1)
    )

    # Return an array of all 4 points using the indices
    # Each point is in its own array of one coordinate
    return [
        polygon[top_left][0],
        polygon[top_right][0],
        polygon[bottom_right][0],
        polygon[bottom_left][0],
    ]


def distance_between(p1, p2):
    """Returns the scalar distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
    """Crops and warp a rectangular section from an image into a square."""

    top_left, top_right, bottom_right, bottom_left = (
        crop_rect[0],
        crop_rect[1],
        crop_rect[2],
        crop_rect[3],
    )

    # Set the data type to float32 or `getPerspectiveTransform` will throw an error
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

    # Get the longest side in the rectangle
    side = max(
        [
            distance_between(bottom_right, top_right),
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),
            distance_between(top_left, top_right),
        ]
    )

    # Describe a square with side of the calculated length,
    # this is the new perspective we want to warp to
    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32"
    )

    # Get the transformation matrix for skewing the image to fit a square
    # by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(img.copy(), m, (int(side), int(side)))


def infer_grid(img):
    """Infers 81 cell grid from a square image."""
    squares = []
    side = img.shape[:1]
    side = side[0] / 9

    # Note that we swap j and i here so the rectangles
    # are stored in the list reading left-right instead of top-down.
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)  # Top left corner of a bounding box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
            squares.append((p1, p2))
    return squares


def cut_from_rectangle(img, rect):
    """Cuts a rectangle from an image using the top left and bottom right points."""
    return img[int(rect[0][1]) : int(rect[1][1]), int(rect[0][0]) : int(rect[1][0])]


def extract_patches(img, squares):
    return [cut_from_rectangle(img, square) for square in squares]


def extract_digit(patch):

    patch = cv2.copyMakeBorder(
        patch,
        2,
        2,
        2,
        2,
        cv2.BORDER_CONSTANT,
    )

    contours, _ = cv2.findContours(
        patch, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    #     print("Nr. contours:", len(contours))

    width = patch.shape[1]
    height = patch.shape[0]

    contours = filter_contour_length(contours)
    # print("After lenght filter:", len(contours))
    contours = filter_center_of_mass(width, height, contours)
    # print("After center filter:", len(contours))
    contours = filter_contours_near_corner(width, height, contours)
    # print("After corner filter:", len(contours))

    if len(contours) == 0:
        return np.zeros((28, 28))

    digit = crop_from_contours(patch, contours)

    #     show_image(digit)

    digit = pad_digit(digit)
    #     show_image(digit)

    digit = resize_digit(digit)
    return digit


def filter_contour_length(contours):
    return [c for c in contours if len(c) > 4]


def filter_center_of_mass(width, height, contours):
    approved_contours = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        if not in_center(width, height, cX, cY):
            continue

        approved_contours.append(c)
    return approved_contours


def in_center(width, height, c_x, c_y, fraction=0.4):
    margin_fraction = (1.0 - fraction) / 2

    width_margin = width * margin_fraction
    in_center_x = width_margin <= c_x <= (width - width_margin)

    height_margin = height * margin_fraction
    in_center_y = height_margin <= c_y <= (height - height_margin)

    return in_center_x and in_center_y


def filter_contours_near_corner(width, height, contours, fraction=1/8):
    accepted_contours = []
    for i, c in enumerate(contours):
        # print("Contour", i)
        x, y, w, h = cv2.boundingRect(c)
        rect = (
            (x, y),
            (x + w, y + h)
        )
        #         image = patch.copy()
        #         image = draw_rects(image, [rect], colour=(0, 255, 0))
        #         show_image(image)

        corners = (
            (x, y),
            (x + w, y),
            (x, y + h),
            (x + w, y + h)
        )

        near_corner = False
        for point in corners:
            if in_corner(width, height, point[0], point[1], fraction):
                # print("Patch rejected: near image corner")
                near_corner = True
                break

        if not near_corner:
            accepted_contours.append(c)

    return accepted_contours


def in_corner(width, height, x, y, fraction=1/8):
    width_margin = int(width * fraction)
    height_margin = int(height * fraction)

    in_top_left = x <= width_margin and y <= height_margin
    in_top_right = x >= width - width_margin and y <= height_margin
    in_bot_left = x <= width_margin and y >= height - height_margin
    in_bot_right = x >= width - width_margin and y >= height - height_margin

    return in_top_left or in_top_right or in_bot_left or in_bot_right


def crop_from_contours(patch, contours):
    #     image = draw_contours(patch.copy(), contours)
    x, y, w, h = cv2.boundingRect(np.concatenate(contours, axis=0))
    #     cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    #     show_image(image)

    rect = (
        (x, y),
        (x + w, y + h)
    )
    return cut_from_rectangle(patch.copy(), rect)


def pad_digit(digit, v_margin=0.1):

    height = digit.shape[0]
    width = digit.shape[1]

    height = max(height, width)

    vertical_margin = int(height * v_margin)
    padded_height = height + 2 * vertical_margin

    horizontal_margin = int((padded_height - width) / 2)

    padded_digit = cv2.copyMakeBorder(
        digit,
        vertical_margin,
        vertical_margin,
        horizontal_margin,
        horizontal_margin,
        cv2.BORDER_CONSTANT,
    )
    return padded_digit


def resize_digit(digit, size=28):
    digit = cv2.resize(digit, (size, size))
    return digit


def scale_and_reshape(img):
    return ((img)/255).reshape(1, 28, 28, 1)


def extract_sudoku(digits, model):
    pred_sudoku = np.zeros((81, 1))
    for i in range(len(digits)):
        if (digits[i]/255).mean()==0.0:
            pred_sudoku[i] = 0
        else:
            img = scale_and_reshape(digits[i])
            pred = model.predict(img)
            pred_sudoku[i] = pred.argmax()
    return pred_sudoku.reshape(9, 9)
