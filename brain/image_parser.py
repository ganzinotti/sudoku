import operator

import cv2
import numpy as np


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def parse_grid(img):
    cropped = crop_to_grid(img)
    enhanced_digits = enhance_digits(cropped)
    digits = extract_digits(enhanced_digits)

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


def extract_digits(img):
    # TODO Make pretty later
    squares = infer_grid(img)
#     patches = get_digits(cropped, squares, 28)    
    patches = extract_patches(img, squares)
    digits = [extract_digit(patch) for patch in patches]
#     pp_digits = [preprocess_digit(extract_feature(digit, 28)) for digit in digits]
    pp_digits = [preprocess_digit(digit) for digit in digits]
    return pp_digits


def extract_patches(img, squares):
    return [cut_from_rectangle(img, square) for square in squares]


def extract_digit(patch):
    corners = find_corners_of_largest_polygon(
        cv2.bitwise_not(patch.copy(), patch.copy())
    )
    cropped = crop_and_warp(patch, corners)
    return cropped


def preprocess_digit(digit, size=28):
    c = 255
    digit = cv2.resize(digit, (size, size))
    return digit

def cut_from_rect(img, rect):
    """Cuts a rectangle from an image using the top left and bottom right points."""
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def scale_and_centre(img, size, margin=0, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
    """
    Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
    connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
    """
    img = inp_img.copy()  # Copy the image, leaving the original untouched
    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    # Loop through the image
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            # Only operate on light or white squares
            if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)
    
    # Colour everything grey (compensates for features outside of our middle scanning range
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

    # Highlight the main feature
    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv2.floodFill(img, mask, (x, y), 0)

            # Find the bounding parameters
            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype='float32'), seed_point


def extract_feature(digit, size):
    """Extracts a digit (if one exists) from a Sudoku square."""

#     digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

    # Use fill feature finding to get the largest feature in middle of the box
    # Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    digit = cut_from_rect(digit, bbox)

    # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    # Ignore any small bounding boxes
    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)


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
