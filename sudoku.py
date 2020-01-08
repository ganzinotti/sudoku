from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__, static_url_path='/static')


@app.route("/")
def index():
    return render_template('index.html', data={})


@app.route("/uploader", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        f = request.files["file"]
        image_path = os.path.join('static', 'images', secure_filename(f.filename))
        f.save(image_path)
        return render_template("index.html", data={'image_path': image_path})


@app.route("/output", methods=["POST"])
def render_output():
    data = request.get_json()["data"]
    return render_template("output.html", data=data)


@app.route("/sample_output", methods=["GET"])
def render_sample_output():
    data = np.random.randint(1, 10, size=(81,))
    return render_template("output.html", data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
