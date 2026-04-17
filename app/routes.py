from flask import Blueprint, render_template, request
import pickle

main = Blueprint("main", __name__)

model = pickle.load(open("models/linear_model.pkl", "rb"))

@main.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        height = float(request.form["height"])
        prediction = model.predict([[height]])[0]

    return render_template("index.html", prediction=prediction)
