from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from brain.solver import solve_sudoku

import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

import numpy as np
import os

app = Flask(__name__, static_url_path='/static')


def retrieve_history_files() -> list:
    return os.listdir(os.path.join('static', 'images'))


def solver_pipeline(image_path: str) -> dict:
    preprocessed_image_path = image_path
    recognized_digits = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ])
    solved_sudoku = solve_sudoku(recognized_digits)
    solved_sudoku = solved_sudoku.flatten().reshape(-1)
    solved_sudoku[6] = 0
    return {
        'image_path': image_path,
        'preprocessed_image_path': preprocessed_image_path,
        'recognized_digits': recognized_digits.flatten().reshape(-1),
        'solved_sudoku': solved_sudoku.flatten().reshape(-1),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" in request.files:
            f = request.files["file"]
            image_path = os.path.join('static', 'images', secure_filename(f.filename))
            f.save(image_path)
            data = solver_pipeline(image_path)
            return render_template("index.html", history_files=retrieve_history_files(), data=data, error=None)
        elif "filename" in request.form:
            filename = request.form["filename"]
            image_path = os.path.join('static', 'images', filename)
            data = solver_pipeline(image_path)
            return render_template("index.html", history_files=retrieve_history_files(), data=data, error=None)
        else:
            return render_template(
                'index.html',
                history_files=retrieve_history_files(),
                data={},
                error="Could not understand request",
            )
    else:
        return render_template('index.html', history_files=retrieve_history_files(), data={}, error=None)


# @app.route("/new_upload", methods=["GET", "POST"])
# def upload_new_file():
#     if request.method == "POST":
#
#
#
# @app.route("/old_file", methods=["GET", "POST"])
# def run_old_file():
#     if request.method == "POST":
#


# @app.route("/output", methods=["POST"])
# def render_output():
#     data = request.get_json()["data"]
#     return render_template("output.html", data=data)
#
#
# @app.route("/sample_output", methods=["GET"])
# def render_sample_output():
#     data = np.random.randint(1, 10, size=(81,))
#     return render_template("output.html", data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
