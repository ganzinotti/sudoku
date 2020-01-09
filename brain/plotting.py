import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_image(img, *args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    ax.imshow(img, interpolation="nearest", cmap="gray")
    plt.tight_layout()
    plt.show()


def plot_many_images(images, titles, rows=1, columns=2):
    """Plots each image in a given list as a grid structure. using Matplotlib."""
    plt.figure(figsize=(9, 9))
    for i, image in enumerate(images):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image, "gray")
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # Hide tick marks
    plt.show()


def draw_points(in_img, points, colour=(255, 0, 0)):
    """Draws circular points on an image."""
    img = in_img.copy()

    radius = int(max(img.shape) / 100)

    img = convert_when_colour(colour, img)

    for point in points:
        img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)

    return img


def draw_rects(in_img, rects, colour=(255, 0, 0)):
    """Displays rectangles on the image."""
    img = convert_when_colour(colour, in_img.copy())

    thickness = int(max(img.shape) / 150)

    for rect in rects:
        img = cv2.rectangle(
            img,
            tuple(int(x) for x in rect[0]),
            tuple(int(x) for x in rect[1]),
            colour,
            thickness,
        )
    return img


def draw_contours(in_img, contours, colour=(255, 0, 0)):
    """Displays contours on the image."""

    img = convert_when_colour(colour, in_img.copy())

    thickness = int(max(img.shape) / 150)

    img = cv2.drawContours(img, contours, -1, colour, thickness)
    return img


def convert_when_colour(colour, img):
    """Dynamically convert an image to colour if the input colour is a
    tuple and the image is grayscale."""
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def image_from_digits(digits, colour=255):
    """Shows list of 81 extracted digits in a grid format"""
    rows = []
    with_border = [
        cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour)
        for img in digits
    ]

    for i in range(9):
        row = np.concatenate(with_border[i * 9: ((i + 1) * 9)], axis=1)
        rows.append(row)

    return np.concatenate(rows)
