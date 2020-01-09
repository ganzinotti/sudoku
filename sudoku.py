from typing import List

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from brain.solver import solve_sudoku
from brain.model.model import load_trained_model

from brain.image_parser import load_image, parse_grid, extract_sudoku
from brain.plotting import image_from_digits

import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

import numpy as np
import os
import cv2

import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, static_url_path="/static")

# set maximum upload to 8MB
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


def retrieve_history_files() -> list:
    return [
        file
        for file in os.listdir(os.path.join("static", "images"))
        if not file.startswith(".")
    ]


def get_digits_image(file_name: str) -> List[np.array]:
    img = load_image(file_name)
    digits = parse_grid(img)

    return digits


def save_preprocessed_image(digits: List[np.array], file_name: str, preprocessed_img_folder: str):
    digit_img = image_from_digits(digits)
    preprocessed_path = os.path.join(preprocessed_img_folder, file_name)
    cv2.imwrite(preprocessed_path, digit_img)

    return preprocessed_path


def solver_pipeline(image_path: str) -> dict:
    logging.info("Start solver")
    digits = get_digits_image(image_path)
    logging.info("Successfully received digits")
    preprocessed_image_path = save_preprocessed_image(
        digits,
        os.path.basename(image_path),
        os.path.join("static", "pp_images"),
    )
    logging.info("Successfully saved preprocessed image")
    model = load_trained_model(os.path.join("brain", "model", "model_weights_custom.hdf5"))
    logging.info("Successfully loaded model")
    recognized_digits = extract_sudoku(digits, model)
    logging.info("Successfully recognized digits")
    solved_sudoku = solve_sudoku(recognized_digits)
    logging.info("Successfully run solver")
    logging.info("Successfully run solver")

    return {
        "image_path": image_path,
        "preprocessed_image_path": preprocessed_image_path,
        "recognized_digits": recognized_digits.flatten().reshape(-1).astype(int),
        "solved_sudoku": solved_sudoku.flatten().reshape(-1),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" in request.files:
            f = request.files["file"]
            image_path = os.path.join("static", "images", secure_filename(f.filename))
            f.save(image_path)
            data = solver_pipeline(image_path)
            return render_template(
                "index.html",
                history_files=retrieve_history_files(),
                data=data,
                error=None,
            )
        elif "filename" in request.form:
            filename = request.form["filename"]
            image_path = os.path.join("static", "images", filename)
            data = solver_pipeline(image_path)
            return render_template(
                "index.html",
                history_files=retrieve_history_files(),
                data=data,
                error=None,
            )
        else:
            return render_template(
                "index.html",
                history_files=retrieve_history_files(),
                data={},
                error="Could not understand request",
            )
    else:
        return render_template(
            "index.html", history_files=retrieve_history_files(), data={}, error=None
        )


@app.route("/flush", methods=["GET"])
def delete_files():
    files = retrieve_history_files()

    logging.info("Delete files.")
    for file_name in files:
        file_path = os.path.join("static", "images", file_name)
        os.remove(file_path)
        logging.info(f"\tDeleted {file_path}")

    return render_template("index.html", history_files=[], data={}, error=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
